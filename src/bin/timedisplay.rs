use ::time::Duration;

pub fn pretty_print_duration(d: Duration) -> String {
    let days = d.num_days();
    let hours = d.num_hours() - days * 24;
    let minutes = d.num_minutes() - days * 24 - hours * 60;
    let seconds = d.num_seconds() - days * 24 - hours * 60 - minutes * 60;

    format!(
        "{d} days {h} hours {m} minutes {s} seconds",
        d = days,
        h = hours,
        m = minutes,
        s = seconds
    )
}
