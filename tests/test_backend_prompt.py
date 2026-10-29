from backend.prompt import build_chat_prompt, extract_reply


def test_prompt_is_a_script_ending_with_the_bot_label():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Good morrow"},
        {"role": "user", "content": "How are you?"},
    ]
    assert build_chat_prompt(messages) == (
        "YOU:\nHello\n\nBOT:\nGood morrow\n\nYOU:\nHow are you?\n\nBOT:\n"
    )


def test_blank_lines_inside_a_message_cannot_fake_the_end_of_a_speech():
    prompt = build_chat_prompt([{"role": "user", "content": "one\n\n\n\ntwo"}])
    assert prompt == "YOU:\none\ntwo\n\nBOT:\n"


def test_long_history_drops_the_oldest_text_but_keeps_the_ending():
    messages = [{"role": "user", "content": "x" * 300}] * 20
    prompt = build_chat_prompt(messages, max_chars=500)
    assert len(prompt) == 500 and prompt.endswith("BOT:\n")


def test_reply_stops_at_a_blank_line():
    assert extract_reply("Good sir, well met.\n\nYOU:\nmore") == "Good sir, well met."


def test_reply_stops_at_the_next_speaker_label():
    assert extract_reply("First line\nSecond line\nROMEO:\nwords") == "First line\nSecond line"


def test_reply_keeps_normal_lines_that_contain_colons():
    assert extract_reply("He said: go\nand left") == "He said: go\nand left"


def test_empty_and_whitespace_replies():
    assert extract_reply("") == ""
    assert extract_reply("\n\nBOT:\nlater") == ""
    assert extract_reply("   \n  ") == ""


def test_reply_stops_at_eos_markers():
    assert extract_reply("Hello there![EOS]extra") == "Hello there!"
    assert extract_reply("Hello there!EOS") == "Hello there!"
    assert extract_reply("Hello there!\n\nEOS") == "Hello there!"
    assert extract_reply("Answer is 42[EOS]\nYOU:\nNext") == "Answer is 42"


def test_reply_preserves_multiline_speech():
    raw = "Line 1.\nLine 2.\nLine 3.\n\nYOU:\nNext question"
    assert extract_reply(raw) == "Line 1.\nLine 2.\nLine 3."


def test_user_mandated_shakespeare_mode():
    """Requirement 1: Shakespeare mode: ROMEO: Hello.\n\nJULIET: Hi. returns only Hello."""
    raw = "ROMEO: Hello.\n\nJULIET: Hi."
    assert extract_reply(raw) == "Hello."


def test_user_mandated_multiline_mode():
    """Requirement 2: Multiline mode: BOT: First paragraph.\n\nSecond paragraph.[EOS] preserves both paragraphs and removes [EOS]."""
    raw = "BOT: First paragraph.\n\nSecond paragraph.[EOS]"
    assert extract_reply(raw) == "First paragraph.\n\nSecond paragraph."


def test_general_speaker_label_detection():
    """General speaker-label detector must identify various names beyond YOU and BOT."""
    assert extract_reply("HAMLET: To be, or not to be.\n\nOPHELIA: My lord?") == "To be, or not to be."
    assert extract_reply("CORIOLANUS: Hear me speak.\n\nFIRST CITIZEN: Peace!") == "Hear me speak."
    assert extract_reply("LADY MACBETH: A deed without a name.\n\nMACBETH: Whence is that knocking?") == "A deed without a name."


def test_names_with_spaces_apostrophes_and_digits():
    """Check 2: Test speaker names with spaces, apostrophes, and digits."""
    # Names with spaces
    assert extract_reply("KING LEAR: Blow, winds!\n\nFOOL: Nuncle, court holy-water.") == "Blow, winds!"
    assert extract_reply("FIRST CITIZEN: Peace!\n\nSECOND CITIZEN: Hear him.") == "Peace!"
    # Names with apostrophes
    assert extract_reply("O'CONNOR: Stand firm.\n\nD'ARTAGNAN: En garde!") == "Stand firm."
    # Names with digits
    assert extract_reply("SOLDIER 1: Who goes there?\n\nSOLDIER 2: A friend.") == "Who goes there?"
    assert extract_reply("AGENT 007: The name is Bond.\n\nBOT: Welcome 007.") == "The name is Bond."


def test_ambiguous_uppercase_lines_in_prose_are_not_mistaken_for_speakers():
    """Check 2: An uppercase line ending in a colon (NOTE:, WARNING:, STEP 1:) must not truncate valid prose."""
    # NOTE: inside a reply
    text = "Here is the guidance.\n\nNOTE: Remember to save the checkpoint.\n\nThen evaluate.[EOS]"
    assert extract_reply(text) == "Here is the guidance.\n\nNOTE: Remember to save the checkpoint.\n\nThen evaluate."

    # STEP 1: / STEP 2: inside a reply
    text2 = "Follow these steps:\n\nSTEP 1: Run preprocessing.\n\nSTEP 2: Train the model.[EOS]"
    assert extract_reply(text2) == "Follow these steps:\n\nSTEP 1: Run preprocessing.\n\nSTEP 2: Train the model."

    # WARNING: / IMPORTANT: inside a reply
    text3 = "BOT: Process completed.\n\nWARNING: Low GPU memory.\n\nIMPORTANT: Use small batch size.[EOS]"
    assert extract_reply(text3) == "Process completed.\n\nWARNING: Low GPU memory.\n\nIMPORTANT: Use small batch size."


def test_ordinary_prose_containing_colons():
    """Check 2: Ordinary prose with colons must remain fully intact."""
    assert extract_reply("He said: go\nand left") == "He said: go\nand left"
    assert extract_reply("Aspect ratio: 16:9\nResolution: 1080p") == "Aspect ratio: 16:9\nResolution: 1080p"
    assert extract_reply("The meeting is at 10:30 AM: please be punctual.") == "The meeting is at 10:30 AM: please be punctual."
    assert extract_reply("The final score was 3:2 in overtime.") == "The final score was 3:2 in overtime."
