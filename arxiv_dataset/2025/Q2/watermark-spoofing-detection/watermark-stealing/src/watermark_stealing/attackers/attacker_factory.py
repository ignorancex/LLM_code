from watermark_stealing.attackers.base_attacker import BaseAttacker
from watermark_stealing.attackers.our_attacker import OurAttacker
from watermark_stealing.attackers.sadasivan_attacker import SadasivanAttacker
from watermark_stealing.config import AttackerAlgo, WsConfig


def get_attacker(cfg: WsConfig) -> BaseAttacker:
    if cfg.attacker.algo == AttackerAlgo.OUR:
        return OurAttacker(
            cfg.meta,
            cfg.attacker,
            cfg.server.watermark.scheme,
            cfg.server.watermark.generation.seeding_scheme,
        )
    elif cfg.attacker.algo == AttackerAlgo.SADASIVAN:
        return SadasivanAttacker(cfg.meta, cfg.attacker)
    else:
        raise ValueError(f"Unknown attacker algo: {cfg.attacker.algo}")
