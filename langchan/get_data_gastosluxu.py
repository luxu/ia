import pandas as pd
from decouple import config
from sqlalchemy import create_engine

engine = create_engine(
    config('DATABASE_URL')
)

def fetch_data(query):
    """Extract data from the database."""
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def load_df():
    """Load data from the database."""
    sql = """
    select gg.id as id_gasto, gp.gasto_id as id_parcela, name, datagasto, gp.valor_parcela, gp.numero_parcela 
    from gasto_gasto gg 
    join gasto_parcelas gp on gp.gasto_id = gg.id 
    group by gp.gasto_id, gg.id, gp.valor_parcela, gp.numero_parcela
    order by gg.id desc
    """
    return fetch_data(sql)


def manipule_df():
    """Manipulate the dataframe."""
    df = load_df()
    # replace the ',' to ',' in the column('valor_parcela')
    # df['valor_parcela'] = df['valor_parcela'].str.replace('.', '').str.replace(',', '.')
    # transform the column('valor_parcela') in the type str to float
    # df.apply(lambda x: float(x))
    # save to filename 'gastos_luxu.csv'
    df.to_csv('gastos_luxu.csv', index=False)

if __name__ == '__main__':
    """Create the csv file."""
    manipule_df()
