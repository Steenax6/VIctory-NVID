//! Switches that are on only with `=1`, read once.

/// Whether the switch `$name` is on: its variable is `1`. With `VICTORY_AB`
/// naming it (diagnostics builds, `crate::ab`) it is on and off while the
/// title runs; otherwise, and in every other build, read once.
macro_rules! toggled {
    ($name:literal) => {{
        #[cfg(feature = "diagnostics")]
        {
            static BIT: std::sync::OnceLock<Result<bool, u32>> = std::sync::OnceLock::new();
            match BIT.get_or_init(|| match $crate::ab::bit($name) {
                Some(bit) => Err(bit),
                None => Ok(std::env::var_os($name).is_some_and(|v| v == "1")),
            }) {
                Ok(on) => *on,
                Err(bit) => $crate::ab::on(*bit),
            }
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var_os($name).is_some_and(|v| v == "1"))
        }
    }};
}
pub(crate) use toggled;
