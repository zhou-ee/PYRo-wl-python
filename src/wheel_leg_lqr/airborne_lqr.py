"""Public API and CLI for the contact-free two-hip LQR schedule."""

# Integration-only mapping from the existing fourteen-state estimator.  The
# airborne plant, gain computation, and schedule never read those states.
FULL_STATE_INDICES = (10, 11, 12, 13, 4, 5)

from .airborne.schedule import (  # noqa: F401
    AirborneGainNode,
    AirborneScheduleFit,
    compute_airborne_gain_node,
    compute_airborne_schedule,
    evaluate_airborne_gain,
    main,
    render_airborne_header,
    render_airborne_summary,
    save_airborne_schedule,
)


if __name__ == "__main__":
    main()
