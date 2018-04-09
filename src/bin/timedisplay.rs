use time::Duration;

pub fn pretty_print_duration(d: Duration) -> String {
    let days = d.num_days();
    let hours = d.num_hours() - days * 24;
    let minutes = d.num_minutes() - days * 24 * 60 - hours * 60;
    let seconds = d.num_seconds() - days * 24 * 60 * 60 - hours * 60 * 60 - minutes * 60;

    format!(
        "{d} days {h} hours {m} minutes {s} seconds",
        d = days,
        h = hours,
        m = minutes,
        s = seconds
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretty_print_duration() {
        // all vaues in seconds
        let d =
            Duration::days(5) + Duration::hours(14) + Duration::minutes(57) + Duration::seconds(14);
        assert_eq!(
            "5 days 14 hours 57 minutes 14 seconds",
            pretty_print_duration(d)
        );

        let d =
            Duration::days(0) + Duration::hours(25) + Duration::minutes(57) + Duration::seconds(14);
        assert_eq!(
            "1 days 1 hours 57 minutes 14 seconds",
            pretty_print_duration(d)
        );

        let d = Duration::seconds(12345678);
        assert_eq!(
            "142 days 21 hours 21 minutes 18 seconds",
            pretty_print_duration(d)
        );
    }

}
