from types import SimpleNamespace as N
import unittest

from scripts.streaming_transport import assemble_completion


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        return iter(self.chunks)


class TestStreamingTransport(unittest.TestCase):
    def test_complete_content_and_final_usage_are_preserved(self):
        usage = N(prompt_tokens=100, completion_tokens=12, total_tokens=112)
        stream = Stream([
            N(choices=[N(index=0, delta=N(content='{"label":'), finish_reason=None)], usage=None),
            N(choices=[N(index=0, delta=N(content='0}'), finish_reason='stop')], usage=None),
            N(choices=[], usage=usage),
        ])
        result = assemble_completion(stream)
        self.assertEqual(result.choices[0].message.content, '{"label":0}')
        self.assertIs(result.usage, usage)
        self.assertTrue(stream.closed)

    def test_partial_stream_cannot_become_a_valid_answer(self):
        stream = Stream([N(choices=[N(index=0, delta=N(content='{"label":0'), finish_reason=None)], usage=None)])
        with self.assertRaises(ValueError):
            assemble_completion(stream)
        self.assertTrue(stream.closed)


if __name__ == '__main__':
    unittest.main()
