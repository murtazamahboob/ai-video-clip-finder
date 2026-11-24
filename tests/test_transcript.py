"""Tests: transcript parsing, sentence boundaries, SRT/VTT round-trip."""

from clipfinder.transcript import (Transcript, Word, parse_srt, parse_vtt,
                                   words_to_srt)


def _tr():
    return Transcript([
        Word("Hello", 0.0, 0.4), Word("world.", 0.4, 0.8),
        Word("How", 1.0, 1.3), Word("are", 1.3, 1.6), Word("you?", 1.6, 2.0),
        Word("Fine.", 2.5, 2.9),
    ])


def test_sentences_split_on_punctuation():
    assert len(_tr().sentences()) == 3


def test_sentence_spans():
    spans = _tr().sentence_spans()
    assert spans[0] == (0.0, 0.8, "Hello world.")
    assert spans[1][2] == "How are you?"


def test_words_in_range():
    assert len(_tr().words_in_range(0.9, 2.1)) == 3
    assert _tr().words_in_range(5, 6) == []


def test_duration_and_text():
    tr = _tr()
    assert tr.duration == 2.9
    assert tr.text.startswith("Hello world.")


def test_parse_srt():
    srt = ("1\n00:00:00,000 --> 00:00:02,000\nHello world.\n\n"
           "2\n00:00:02,500 --> 00:00:04,000\nHow are you?\n")
    tr = parse_srt(srt)
    assert len(tr.words) == 5
    assert tr.sentences()[0][0].start == 0.0
    assert tr.words[-1].end == 4.0


def test_parse_vtt_skips_header():
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello world.\n"
    tr = parse_vtt(vtt)
    assert len(tr.words) == 2
    assert tr.words[0].text == "Hello"


def test_words_to_srt_round_trip():
    srt = words_to_srt(_tr().words)
    assert "Hello world." in srt
    assert parse_srt(srt).text == _tr().text


def test_empty_transcript():
    tr = Transcript([])
    assert tr.duration == 0.0
    assert tr.sentences() == []
