from support_agent.pii import redact


def test_redacts_email_phone_card_pan_aadhaar():
    text = (
        "Mail me at rahul.k@example.com or call +91 98765 43210. "
        "Card 4111 1111 1111 1111, PAN ABCDE1234F, Aadhaar 1234 5678 9012."
    )
    r = redact(text)
    assert "example.com" not in r.text
    assert "98765" not in r.text
    assert "4111" not in r.text
    assert "ABCDE1234F" not in r.text
    assert "1234 5678 9012" not in r.text
    assert r.found == {"EMAIL": 1, "PHONE": 1, "CARD": 1, "PAN": 1, "AADHAAR": 1}


def test_leaves_normal_numbers_alone():
    r = redact("My order 12345 failed after 3 days, I paid Rs 1499")
    assert r.text == "My order 12345 failed after 3 days, I paid Rs 1499"
    assert r.found == {}


def test_non_luhn_long_number_is_not_called_a_card():
    r = redact("ref 1234 5678 9012 3456")
    assert "[CARD]" not in r.text
