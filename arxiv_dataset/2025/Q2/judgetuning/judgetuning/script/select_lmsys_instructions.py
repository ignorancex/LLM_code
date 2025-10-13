from sklearn.model_selection import train_test_split

from judgetuning.utils import download_hf
from judgetuning.judge_paths import judge_tuning_datadir
import pandas as pd


def select_lmsys_instructions(min_criterion: int = 5) -> list[str]:
    instr_quality_path = (
        judge_tuning_datadir / "instruction-quality" / "lmsys-instruction-quality.csv"
    )
    if not instr_quality_path.exists():
        print(f"Local files for {instr_quality_path} not found, downloading from HF.")
        download_hf(name="instruction-quality", local_path=judge_tuning_datadir)

    df_lmsys = pd.read_csv(instr_quality_path)
    df_lmsys.drop_duplicates(subset=["instruction"], inplace=True)

    # Keep instructions that match criterion "1" (non ambiguity) and have a quality of at least 5 => 6548 instructions
    mask = df_lmsys.apply(
        lambda row: "1" in row["criterions"] and row["n_criterions"] >= min_criterion,
        axis=1,
    )
    return df_lmsys.loc[mask, "instruction"].tolist()


def val_test_split() -> tuple[list[str], list[str]]:
    instructions = select_lmsys_instructions()

    train_instrs, test_instrs = train_test_split(
        instructions, test_size=3000, random_state=42
    )
    return train_instrs, test_instrs


if __name__ == "__main__":
    instructions = select_lmsys_instructions()
    print(len(instructions))

    train_instrs, test_instrs = val_test_split()
    print(len(train_instrs), len(test_instrs))

    print("\nfirst train instr\n", train_instrs[0])
    print("\nfirst test instr\n", test_instrs[0])
