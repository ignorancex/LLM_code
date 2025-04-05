from .Average import Average

from .CooMed import CooMed, TrMean
from .GeoMed import GeoMed
from .Krum import Krum, MultiKrum

from .Bucket import BucketKrum, BucketMultiKrum
from .RAGE import RAGE

from .Reject import SelfReject, AverageReject, Zeno
from .FLTrust import FLTrust
from .ByGARS import ByGARS

from .BOBA import BOBA, BOBA_ES, BOBA_No_Stage1, BOBA_No_Stage2




def create_defense(args):
    if args.defense == 'average':
        defense = Average(args)

    elif args.defense == 'coomed':
        defense = CooMed(args)
    elif args.defense == 'trmean':
        defense = TrMean(args)
    elif args.defense == 'geomed':
        defense = GeoMed(args)
    elif args.defense == 'krum':
        defense = Krum(args)
    elif args.defense == 'mkrum':
        defense = MultiKrum(args)

    elif args.defense == 'bkrum':
        defense = BucketKrum(args)
    elif args.defense == 'bmkrum':
        defense = BucketMultiKrum(args)

    elif args.defense == 'rage':
        defense = RAGE(args)

    elif args.defense == 'selfrej':
        defense = SelfReject(args)
    elif args.defense == 'avgrej':
        defense = AverageReject(args)
    elif args.defense == 'zeno':
        defense = Zeno(args)

    elif args.defense == 'fltrust':
        defense = FLTrust(args)

    elif args.defense == 'bygars':
        defense = ByGARS(args)

    elif args.defense == 'boba':
        defense = BOBA(args)
    elif args.defense == 'boba_es':
        defense = BOBA_ES(args)
    elif args.defense == 'boba_no_stage1':
        defense = BOBA_No_Stage1(args)
    elif args.defense == 'boba_no_stage2':
        defense = BOBA_No_Stage2(args)

    else:
        raise NotImplementedError

    return defense
