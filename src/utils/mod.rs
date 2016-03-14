/// Defines a lower and an upper range.
struct Range {
    lower: f64,
    upper: f64,
}

/// Calculates histogram of `nbins` bins of given numeric vector `samples`.
///
/// # Examples
///
/// ```
/// let x = stochasticsampling::random::NormalDistributionIterator;
/// let samples = x.take(5000).collect::<Vec<_>>();
///
/// let hist = histogram(
///     Range{ lower: -5., upper: 5.},
///     40,
///     &samples
///     );
///
/// for n in hist {
///     println!("{}", n);
/// }
/// ```
pub fn histogram(range: Range, nbins: usize, samples: &Vec<f64>) -> Vec<usize> {
    let mut hist = Vec::with_capacity(nbins);

    let stepsize = (range.upper - range.lower) / (nbins as f64);

    for i in (0..nbins) {
        let fi = i as f64;
        let a = range.lower + fi * stepsize;
        let b = range.lower + (fi + 1.) * stepsize;

        hist.push(
            samples.into_iter()
                .filter(|&x| (a <= *x) && (*x < b))
                .count()
            )
    }

    hist
}
