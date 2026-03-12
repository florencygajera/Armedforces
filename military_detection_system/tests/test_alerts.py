from src.alerts import AlertSystem


def test_trigger_alert_returns_payload_and_rate_limits():
    alert_system = AlertSystem(alert_cooldown_sec=60)

    # Avoid side effects during tests
    alert_system.play_alert_sound = lambda *args, **kwargs: None
    alert_system.client.publish = lambda *args, **kwargs: None

    payload = alert_system.trigger_alert("drone", 0.92, "cctv_1")
    assert payload is not None
    assert payload["class"] == "drone"
    assert payload["status"] == "suspicious_activity_detected"
    assert "Evacuate" in payload["message"]

    # same camera + class should be throttled by cooldown
    second_payload = alert_system.trigger_alert("drone", 0.95, "cctv_1")
    assert second_payload is None


def test_rate_limit_is_per_camera_and_class():
    alert_system = AlertSystem(alert_cooldown_sec=60)

    first = alert_system._is_rate_limited("drone", "cctv_1")
    second_same_key = alert_system._is_rate_limited("drone", "cctv_1")
    different_camera = alert_system._is_rate_limited("drone", "cctv_2")
    different_class = alert_system._is_rate_limited("tank", "cctv_1")

    assert first is False
    assert second_same_key is True
    assert different_camera is False
    assert different_class is False
