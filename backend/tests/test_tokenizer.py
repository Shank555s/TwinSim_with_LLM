import pytest

from llm.tokenizer import TokenizerError, TwinSimTokenizer

TEXTS = ["I would support the policy because it helps families.",
         "Honestly, I am not sure about this at all!",
         "The council should fund public transit and schools."] * 20


def test_special_tokens_present():
    tok = TwinSimTokenizer.train(TEXTS, vocab_size=300)
    assert tok.decode([tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id]).count("<") >= 4
    assert len({tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id}) == 4


def test_roundtrip_including_unicode():
    tok = TwinSimTokenizer.train(TEXTS, vocab_size=320)
    for s in ["I would support the policy.", "naïve café — 😀", "  spaces\tand\nnewlines "]:
        assert tok.decode(tok.encode(s)) == s


def test_bos_eos_and_vocab_size():
    tok = TwinSimTokenizer.train(TEXTS, vocab_size=300)
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.vocab_size <= 300
    assert max(tok.encode(" ".join(TEXTS))) < tok.vocab_size


def test_merges_compress():
    tok = TwinSimTokenizer.train(TEXTS, vocab_size=400)
    s = TEXTS[0]
    assert len(tok.encode(s)) < len(s.encode("utf-8"))


def test_save_load_and_version(tmp_path):
    tok = TwinSimTokenizer.train(TEXTS, vocab_size=300)
    tok.save(tmp_path / "tok.json")
    tok2 = TwinSimTokenizer.load(tmp_path / "tok.json")
    assert tok2.version == tok.version
    assert tok2.encode(TEXTS[1]) == tok.encode(TEXTS[1])


def test_vocab_too_small():
    with pytest.raises(TokenizerError):
        TwinSimTokenizer.train(TEXTS, vocab_size=10)
