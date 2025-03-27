
from splogic.seq2seq.validation import DenotationEqual
from kqapro.evaluate import whether_equal

from dhnamlib.pylib.klass import subclass, implement


@subclass
class KQAProDenotationEqual(DenotationEqual):
    @implement
    def __call__(self, prediction, answer):
        return whether_equal(answer=answer, prediction=prediction)
