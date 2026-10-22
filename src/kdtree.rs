//! A KD-tree for fixed-radius searches, and a brute-force searcher with the same
//! interface.

use crate::metric::{Distance, Thresholds};
use crate::points::Points;
use crate::scalar::Scalar;

/// Leaves hold at most this many points. Benchmarked: smaller leaves prune better in
/// low dimensions, larger ones cost less traversal in high dimensions.
pub const LEAF_SIZE: usize = 32;

/// Something that can find all stored points within the tolerance of a query.
pub trait Searcher<T: Scalar>: Sync {
    /// Whether the matches of one query are emitted in increasing index order.
    const SORTED: bool;

    /// Calls `emit(index, distance)` for every stored point within `th.tol` of `q`.
    /// `index` is the point's row in the original input. The distance is only
    /// computed when `want_dist` is set (otherwise it may be zero).
    fn search<M: Distance<T>>(
        &self,
        m: &M,
        q: &[T],
        th: &Thresholds<T>,
        want_dist: bool,
        stack: &mut Vec<usize>,
        emit: &mut impl FnMut(usize, T),
    );

    /// The number of distinct [`Searcher::locality_key`] values (0: no keys).
    fn n_locality_keys(&self) -> usize {
        0
    }

    /// A key in `0..n_locality_keys()` such that queries with nearby keys visit
    /// mostly the same data, so searching them one after another hits the cache.
    fn locality_key(&self, _q: &[T]) -> usize {
        0
    }

    /// The keys of the queries `start..start + out.len()`.
    fn locality_keys(&self, queries: &Points<T>, start: usize, out: &mut [usize]) {
        for (k, key) in out.iter_mut().enumerate() {
            *key = self.locality_key(queries.row(start + k));
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Node {
    start: usize,
    end: usize,
    /// Index of the left child; the right child is `left + 1`. 0 marks a leaf (the
    /// root has index 0, so it is never a child).
    left: usize,
    /// The dimension the node is split on (internal nodes only).
    dim: usize,
}

/// A balanced KD-tree. Points are stored contiguously in tree order, so a leaf scan
/// reads one block of memory, and every node keeps the bounding box of its points.
#[derive(Clone, Debug)]
pub struct KdTree<T> {
    d: usize,
    data: Vec<T>,
    idx: Vec<usize>,
    nodes: Vec<Node>,
    /// Per node: `d` lower bounds, then `d` upper bounds.
    bounds: Vec<T>,
    /// Per node: the smallest value in dimension `dim` of the right child (internal
    /// nodes only).
    splits: Vec<T>,
    /// Per node: the largest value in dimension `dim` of the left child (internal
    /// nodes only).
    lmax: Vec<T>,
}

/// Subtrees with fewer points than this are built on the current thread.
const PARALLEL_BUILD_MIN: usize = 32_768;

/// Whether a (sub)tree of `n` points is split on two threads.
fn split_in_parallel(n: usize, n_threads: usize) -> bool {
    n_threads > 1 && n >= PARALLEL_BUILD_MIN
}

/// The nodes of a (sub)tree under construction. Node 0 is its root.
///
/// The build is top-down with *loose* boxes: a child's box is its parent's box
/// clipped at the split value, which costs nothing. The split dimension is chosen
/// from the loose box. Afterwards every box is tightened bottom-up: leaves from their
/// points (one pass over all points), internal nodes as the union of their children.
struct Builder<T> {
    d: usize,
    nodes: Vec<Node>,
    bounds: Vec<T>,
    splits: Vec<T>,
    lmax: Vec<T>,
}

impl<T: Scalar> Builder<T> {
    /// Builds the tree over `perm` (point indices, reordered in place), whose first
    /// element is at position `base` of the whole tree. `root_box` is the (loose)
    /// box of the root; without it, the root's box is computed from the points.
    fn build(
        points: &Points<T>,
        perm: &mut [usize],
        base: usize,
        leaf_size: usize,
        n_threads: usize,
        root_box: Option<&[T]>,
    ) -> Self {
        let mut b = Self {
            d: points.dims(),
            nodes: Vec::with_capacity(2 * perm.len().div_ceil(leaf_size) + 1),
            bounds: Vec::new(),
            splits: Vec::new(),
            lmax: Vec::new(),
        };
        match root_box {
            Some(bx) => b.add_node(bx, base, perm.len()),
            None => {
                let bx = bbox(points, perm);
                b.add_node(&bx, base, perm.len())
            }
        };
        b.split(points, perm, 0, leaf_size, n_threads);
        if !split_in_parallel(perm.len(), n_threads) {
            // A parallel split grafts tightened subtrees and unions them itself.
            b.tighten(points, perm, base);
        }
        b
    }

    /// Adds a node for `n` points at position `start` with the box `bx`.
    fn add_node(&mut self, bx: &[T], start: usize, n: usize) -> usize {
        self.bounds.extend_from_slice(bx);
        self.nodes.push(Node {
            start,
            end: start + n,
            left: 0,
            dim: 0,
        });
        self.splits.push(T::ZERO);
        self.lmax.push(T::ZERO);
        self.nodes.len() - 1
    }

    /// The box of node `id`.
    fn bx(&self, id: usize) -> &[T] {
        &self.bounds[id * 2 * self.d..(id + 1) * 2 * self.d]
    }

    /// Adds a child of `parent` whose box is the parent's clipped at `split` in
    /// `dim`: the upper bound for the left child, the lower bound for the right one.
    fn add_loose(&mut self, parent: usize, start: usize, n: usize, split: T, left: bool) -> usize {
        let d = self.d;
        let dim = self.nodes[parent].dim;
        let at = self.bounds.len();
        self.bounds
            .extend_from_within(parent * 2 * d..(parent + 1) * 2 * d);
        self.bounds[at + if left { d + dim } else { dim }] = split;
        self.nodes.push(Node {
            start,
            end: start + n,
            left: 0,
            dim: 0,
        });
        self.splits.push(T::ZERO);
        self.lmax.push(T::ZERO);
        self.nodes.len() - 1
    }

    /// Splits node `id` (holding the points `perm`) at the median of its widest
    /// dimension, recursively. Large subtrees are built on separate threads.
    fn split(
        &mut self,
        points: &Points<T>,
        perm: &mut [usize],
        id: usize,
        leaf_size: usize,
        n_threads: usize,
    ) {
        let n = perm.len();
        if n <= leaf_size {
            return;
        }
        let d = self.d;
        let (lo, hi) = self.bx(id).split_at(d);
        let mut dim = 0;
        let mut widest = T::ZERO;
        for k in 0..d {
            let spread = hi[k] - lo[k];
            if spread > widest {
                widest = spread;
                dim = k;
            }
        }
        if widest <= T::ZERO {
            // All points are identical (or d == 0): splitting cannot help.
            return;
        }
        let values = points.data();
        let key = |i: usize| values[i * d + dim];
        let mid = n / 2;
        perm.select_nth_unstable_by(mid, |&a, &b| {
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let split = key(perm[mid]);
        self.nodes[id].dim = dim;
        self.splits[id] = split;
        let base = self.nodes[id].start;
        let (left_perm, right_perm) = perm.split_at_mut(mid);
        if split_in_parallel(n, n_threads) {
            let right_threads = n_threads / 2;
            let mut left_box = self.bx(id).to_vec();
            let mut right_box = left_box.clone();
            left_box[d + dim] = split;
            right_box[dim] = split;
            let (left, right) = rayon::join(
                || {
                    Self::build(
                        points,
                        left_perm,
                        base,
                        leaf_size,
                        n_threads - right_threads,
                        Some(&left_box),
                    )
                },
                || {
                    Self::build(
                        points,
                        right_perm,
                        base + mid,
                        leaf_size,
                        right_threads,
                        Some(&right_box),
                    )
                },
            );
            // Both subtrees are tight; this node is their union.
            for k in 0..d {
                self.bounds[id * 2 * d + k] = left.bounds[k].min(right.bounds[k]);
                self.bounds[id * 2 * d + d + k] = left.bounds[d + k].max(right.bounds[d + k]);
            }
            self.lmax[id] = left.bounds[d + dim];
            self.graft(id, left, right);
        } else {
            let left = self.add_loose(id, base, mid, split, true);
            self.add_loose(id, base + mid, n - mid, split, false);
            self.nodes[id].left = left;
            self.split(points, left_perm, left, leaf_size, 1);
            self.split(points, right_perm, left + 1, leaf_size, 1);
        }
    }

    /// Replaces the loose boxes by the bounding boxes of the points: leaves from
    /// their points, internal nodes as the union of their children. Children always
    /// come after their parent in `nodes`, so one pass in reverse order suffices.
    fn tighten(&mut self, points: &Points<T>, perm: &[usize], base: usize) {
        let d = self.d;
        for id in (0..self.nodes.len()).rev() {
            let node = self.nodes[id];
            if node.left == 0 {
                let bx = bbox(points, &perm[node.start - base..node.end - base]);
                self.bounds[id * 2 * d..(id + 1) * 2 * d].copy_from_slice(&bx);
            } else {
                let (l, r) = (node.left, node.left + 1);
                for k in 0..d {
                    let lo = self.bounds[l * 2 * d + k].min(self.bounds[r * 2 * d + k]);
                    let hi = self.bounds[l * 2 * d + d + k].max(self.bounds[r * 2 * d + d + k]);
                    self.bounds[id * 2 * d + k] = lo;
                    self.bounds[id * 2 * d + d + k] = hi;
                }
                self.lmax[id] = self.bounds[l * 2 * d + d + node.dim];
            }
        }
    }

    /// Attaches two separately built subtrees as the children of node `id`. The two
    /// roots must be adjacent, so the order is: both roots, then the rest of the left
    /// subtree, then the rest of the right one.
    fn graft(&mut self, id: usize, left: Self, right: Self) {
        let off = self.nodes.len();
        let n_left = left.nodes.len();
        let map_left = |k: usize| if k == 0 { off } else { off + 1 + k };
        let map_right = |k: usize| if k == 0 { off + 1 } else { off + n_left + k };
        let d2 = 2 * self.d;
        let mut push = |b: &Self, k: usize, map: &dyn Fn(usize) -> usize| {
            let mut node = b.nodes[k];
            if node.left != 0 {
                node.left = map(node.left);
            }
            self.nodes.push(node);
            self.bounds
                .extend_from_slice(&b.bounds[k * d2..(k + 1) * d2]);
            self.splits.push(b.splits[k]);
            self.lmax.push(b.lmax[k]);
        };
        push(&left, 0, &map_left);
        push(&right, 0, &map_right);
        for k in 1..n_left {
            push(&left, k, &map_left);
        }
        for k in 1..right.nodes.len() {
            push(&right, k, &map_right);
        }
        self.nodes[id].left = off;
    }
}

/// The bounding box of the points `perm` (non-empty): `d` lower, then `d` upper bounds.
fn bbox<T: Scalar>(points: &Points<T>, perm: &[usize]) -> Vec<T> {
    let d = points.dims();
    let mut bx = Vec::with_capacity(2 * d);
    bx.extend_from_slice(points.row(perm[0]));
    bx.extend_from_slice(points.row(perm[0]));
    let (lo, hi) = bx.split_at_mut(d);
    for &i in &perm[1..] {
        for (k, &v) in points.row(i).iter().enumerate() {
            lo[k] = lo[k].min(v);
            hi[k] = hi[k].max(v);
        }
    }
    bx
}

impl<T: Scalar> KdTree<T> {
    /// Builds the tree on up to `n_threads` threads. Takes ownership of the points,
    /// whose memory is released once they are copied into tree order.
    pub fn build(points: Points<T>, n_threads: usize) -> Self {
        Self::with_leaf_size(points, LEAF_SIZE, n_threads)
    }

    pub fn with_leaf_size(points: Points<T>, leaf_size: usize, n_threads: usize) -> Self {
        let n = points.len();
        let d = points.dims();
        let mut perm: Vec<usize> = (0..n).collect();
        let (nodes, bounds, splits, lmax) = if n > 0 {
            let b = Builder::build(&points, &mut perm, 0, leaf_size.max(1), n_threads, None);
            (b.nodes, b.bounds, b.splits, b.lmax)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        // Copy into tree order. (Permuting in place would save this copy, but chasing
        // the permutation's cycles is several times slower.) The original order is
        // freed right away, so it is not kept alive during the search.
        let mut data = Vec::with_capacity(n * d);
        for &i in &perm {
            data.extend_from_slice(points.row(i));
        }
        let (_, orig) = points.into_parts();
        Self {
            d,
            data,
            idx: perm.iter().map(|&i| orig[i]).collect(),
            nodes,
            bounds,
            splits,
            lmax,
        }
    }

    /// The number of stored points.
    pub fn len(&self) -> usize {
        self.idx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    #[inline(always)]
    fn row(&self, pos: usize) -> &[T] {
        &self.data[pos * self.d..(pos + 1) * self.d]
    }

    /// Searches the subtree under `root`, pruning with the bounding boxes.
    #[allow(clippy::too_many_arguments)]
    fn search_from<M: Distance<T>>(
        &self,
        root: usize,
        m: &M,
        q: &[T],
        th: &Thresholds<T>,
        want_dist: bool,
        stack: &mut Vec<usize>,
        emit: &mut impl FnMut(usize, T),
    ) {
        let d = self.d;
        stack.clear();
        stack.push(root);
        while let Some(id) = stack.pop() {
            let node = self.nodes[id];
            let (lo, hi) = self.bounds[id * 2 * d..(id + 1) * 2 * d].split_at(d);
            let lower = m.box_min(q, lo, hi);
            if lower > th.prune {
                continue;
            }
            if lower <= th.accept && m.box_max(q, lo, hi) <= th.accept {
                // Every point in the box is within the tolerance.
                for pos in node.start..node.end {
                    let dist = if want_dist {
                        let x = self.row(pos);
                        m.finish(q, x, m.reduced(q, x))
                    } else {
                        T::ZERO
                    };
                    emit(self.idx[pos], dist);
                }
                continue;
            }
            if node.left == 0 {
                for pos in node.start..node.end {
                    let x = self.row(pos);
                    let r = m.reduced(q, x);
                    if r <= th.prune {
                        let dist = m.finish(q, x, r);
                        if dist <= th.tol {
                            emit(self.idx[pos], dist);
                        }
                    }
                }
            } else {
                stack.push(node.left + 1);
                stack.push(node.left);
            }
        }
    }
}

impl<T: Scalar> Searcher<T> for KdTree<T> {
    const SORTED: bool = false;

    /// Walks down the child that contains `q` by the split values alone. At every
    /// level the other child is checked with a one-dimensional bound (its distance
    /// to `q` along the split dimension is a lower bound of its box distance) and
    /// searched with its boxes only if it may hold a match. This visits about two
    /// nodes per level only where needed instead of everywhere.
    fn search<M: Distance<T>>(
        &self,
        m: &M,
        q: &[T],
        th: &Thresholds<T>,
        want_dist: bool,
        stack: &mut Vec<usize>,
        emit: &mut impl FnMut(usize, T),
    ) {
        if self.nodes.is_empty() {
            return;
        }
        let mut id = 0;
        loop {
            let node = self.nodes[id];
            if node.left == 0 {
                self.search_from(id, m, q, th, want_dist, stack, emit);
                return;
            }
            let dim = node.dim;
            let qk = &q[dim..dim + 1];
            if q[dim] < self.splits[id] {
                if m.box_min(qk, &[self.splits[id]], &[T::INFINITY]) <= th.prune {
                    self.search_from(node.left + 1, m, q, th, want_dist, stack, emit);
                }
                id = node.left;
            } else {
                if m.box_min(qk, &[-T::INFINITY], &[self.lmax[id]]) <= th.prune {
                    self.search_from(node.left, m, q, th, want_dist, stack, emit);
                }
                id = node.left + 1;
            }
        }
    }

    fn n_locality_keys(&self) -> usize {
        self.len()
    }

    /// The first position of the leaf that `q` falls into. Leaves are laid out in
    /// tree order, so nearby keys belong to nearby leaves.
    fn locality_key(&self, q: &[T]) -> usize {
        let mut id = 0;
        loop {
            let node = self.nodes[id];
            if node.left == 0 {
                return node.start;
            }
            id = if q[node.dim] < self.splits[id] {
                node.left
            } else {
                node.left + 1
            };
        }
    }

    /// Descends eight queries in lockstep, so that their dependent node loads
    /// overlap instead of waiting for each other.
    fn locality_keys(&self, queries: &Points<T>, start: usize, out: &mut [usize]) {
        const B: usize = 8;
        let mut k = 0;
        while k + B <= out.len() {
            let mut ids = [0usize; B];
            let mut pending = B;
            while pending > 0 {
                pending = 0;
                for (j, id) in ids.iter_mut().enumerate() {
                    let node = self.nodes[*id];
                    if node.left == 0 {
                        continue;
                    }
                    pending += 1;
                    let q = queries.row(start + k + j);
                    *id = if q[node.dim] < self.splits[*id] {
                        node.left
                    } else {
                        node.left + 1
                    };
                }
            }
            for (slot, &id) in out[k..k + B].iter_mut().zip(&ids) {
                *slot = self.nodes[id].start;
            }
            k += B;
        }
        for (j, slot) in out.iter_mut().enumerate().skip(k) {
            *slot = self.locality_key(queries.row(start + j));
        }
    }
}

/// Compares the query against every point. Used by `naive_find_matches`.
#[derive(Clone, Copy, Debug)]
pub struct Brute<'a, T> {
    pub points: &'a Points<T>,
}

impl<T: Scalar> Searcher<T> for Brute<'_, T> {
    const SORTED: bool = true;

    fn search<M: Distance<T>>(
        &self,
        m: &M,
        q: &[T],
        th: &Thresholds<T>,
        _want_dist: bool,
        _stack: &mut Vec<usize>,
        emit: &mut impl FnMut(usize, T),
    ) {
        for i in 0..self.points.len() {
            let x = self.points.row(i);
            let r = m.reduced(q, x);
            if r <= th.prune {
                let dist = m.finish(q, x, r);
                if dist <= th.tol {
                    emit(self.points.orig(i), dist);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Euclidean;

    fn test_points(n: usize, d: usize) -> Points<f64> {
        let values: Vec<f64> = (0..(n * d) as u64)
            .map(|i| ((i * 7919) % 100_003) as f64)
            .collect();
        Points::new(&values, n, d, false)
    }

    /// Every box is exactly the bounding box of its points, and `splits`/`lmax` are
    /// the extremes of the children in the split dimension.
    fn check_boxes(tree: &KdTree<f64>) {
        let d = tree.d;
        for (id, node) in tree.nodes.iter().enumerate() {
            let (lo, hi) = tree.bounds[id * 2 * d..(id + 1) * 2 * d].split_at(d);
            for k in 0..d {
                let col = (node.start..node.end).map(|pos| tree.row(pos)[k]);
                assert_eq!(lo[k], col.clone().fold(f64::INFINITY, f64::min));
                assert_eq!(hi[k], col.fold(-f64::INFINITY, f64::max));
            }
            if node.left != 0 {
                let (l, r) = (tree.nodes[node.left], tree.nodes[node.left + 1]);
                let lmax = (l.start..l.end)
                    .map(|pos| tree.row(pos)[node.dim])
                    .fold(-f64::INFINITY, f64::max);
                let rmin = (r.start..r.end)
                    .map(|pos| tree.row(pos)[node.dim])
                    .fold(f64::INFINITY, f64::min);
                assert_eq!(tree.lmax[id], lmax);
                assert_eq!(tree.splits[id], rmin);
                assert!(lmax <= rmin);
            }
        }
    }

    #[test]
    fn tree_holds_every_point_once() {
        let points = test_points(500, 2);
        let tree = KdTree::with_leaf_size(points.clone(), 4, 1);
        let mut idx = tree.idx.clone();
        idx.sort_unstable();
        assert_eq!(idx, (0..500).collect::<Vec<_>>());
        for (pos, &i) in tree.idx.iter().enumerate() {
            assert_eq!(tree.row(pos), points.row(i));
        }
        // Children cover their parent's range.
        for node in &tree.nodes {
            if node.left != 0 {
                let (l, r) = (tree.nodes[node.left], tree.nodes[node.left + 1]);
                assert_eq!((l.start, l.end, r.end), (node.start, r.start, node.end));
            }
        }
        check_boxes(&tree);
    }

    #[test]
    fn parallel_build_equals_serial_build() {
        // More points than PARALLEL_BUILD_MIN, so the build really splits up.
        let n = 2 * PARALLEL_BUILD_MIN;
        let points = test_points(n, 3);
        let serial = KdTree::with_leaf_size(points.clone(), 8, 1);
        check_boxes(&serial);
        for threads in [2, 3, 8] {
            let parallel = KdTree::with_leaf_size(points.clone(), 8, threads);
            assert_eq!(parallel.idx, serial.idx);
            assert_eq!(parallel.data, serial.data);
            assert_eq!(parallel.nodes.len(), serial.nodes.len());
            check_boxes(&parallel);
            // Same tree, different node numbering: compare it structurally.
            fn walk(t: &KdTree<f64>, id: usize, out: &mut Vec<(usize, usize, usize)>) {
                let n = t.nodes[id];
                out.push((n.start, n.end, n.dim));
                if n.left != 0 {
                    walk(t, n.left, out);
                    walk(t, n.left + 1, out);
                }
            }
            let (mut a, mut b) = (Vec::new(), Vec::new());
            walk(&serial, 0, &mut a);
            walk(&parallel, 0, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn identical_points_stay_in_one_leaf() {
        let values = vec![1.5_f64; 2 * 100];
        let points = Points::new(&values, 100, 2, false);
        let tree = KdTree::with_leaf_size(points.clone(), 4, 1);
        assert_eq!(tree.nodes.len(), 1);
        let th = Distance::<f64>::thresholds(&Euclidean, 0.0, 2);
        let mut found = Vec::new();
        tree.search(
            &Euclidean,
            &[1.5, 1.5],
            &th,
            false,
            &mut Vec::new(),
            &mut |i, _| found.push(i),
        );
        assert_eq!(found.len(), 100);
    }

    #[test]
    fn batched_keys_equal_single_keys() {
        let tree = KdTree::with_leaf_size(test_points(3000, 2), 8, 1);
        let queries = test_points(1003, 2);
        let mut batched = vec![0; queries.len()];
        tree.locality_keys(&queries, 0, &mut batched);
        let single: Vec<usize> = (0..queries.len())
            .map(|i| tree.locality_key(queries.row(i)))
            .collect();
        assert_eq!(batched, single);
    }
}
