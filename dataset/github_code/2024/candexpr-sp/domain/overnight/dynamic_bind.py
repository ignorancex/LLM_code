
from dhnamlib.pylib.klass import subclass, implement

from splogic.seq2seq.dynamic_bind import DynamicBinder


@subclass
class DomainDynamicBinder(DynamicBinder):
    @implement
    def bind_example(self, example, grammar=None):
        binding = dict(domain=example['domain'])
        return binding

    @implement
    def bind_batch(self, batched_example, grammar=None):
        bindings = tuple(
            dict(domain=domain)
            for domain in batched_example['domain']
        )
        return bindings
