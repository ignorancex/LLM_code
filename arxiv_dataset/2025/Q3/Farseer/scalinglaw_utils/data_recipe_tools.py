
import re
import pandas as pd
import typer

def get_data_recipe_df(table_str: str) -> pd.DataFrame:
    pattern = re.compile('│\\s*([^│]+?)\\s*│\\s*([^│]+?)\\s*│\\s*([^│]+?)\\s*│\\s*([^│]+?)\\s*│')
    matches = pattern.findall(table_str)
    columns = [match.strip() for match in matches[0]]
    data = [list(map(str.strip, match)) for match in matches[1:]]
    df = pd.DataFrame(data, columns=columns)
    return df
