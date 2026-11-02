import pandas as pd

df = pd.read_csv('results/period_strategy_matrix.csv')
df = df[df['Strategy'] != 'markowitz']
rows = []
for key, g in df.groupby(['PortfolioCreationDate', 'InvestmentEndDate']):
    for metric, ascending in [('CAGR', False), ('Sharpe', False), ('MaxDrawdown', True)]:
        r = g.sort_values(metric, ascending=ascending)[['Strategy', metric]].reset_index(drop=True).reset_index().rename(columns={'index': 'Rank'})
        r['Rank'] = r['Rank'] + 1
        r['Metric'] = metric
        r['PortfolioCreationDate'] = key[0]
        r['InvestmentEndDate'] = key[1]
        rows.append(r[['PortfolioCreationDate', 'InvestmentEndDate', 'Metric', 'Rank', 'Strategy', metric]])
res = pd.concat(rows, ignore_index=True)
win_counts = res[res['Rank'] == 1].groupby('Strategy')['Metric'].value_counts().unstack(fill_value=0)
print('WIN_COUNTS_BY_STRATEGY_AND_METRIC')
print(win_counts[['CAGR', 'Sharpe', 'MaxDrawdown']].to_string())
print('\nAVG_METRIC_BY_STRATEGY')
avg = df.groupby('Strategy')[['CAGR', 'Sharpe', 'MaxDrawdown']].mean()
print(avg[['CAGR', 'Sharpe', 'MaxDrawdown']].sort_values(['CAGR','Sharpe','MaxDrawdown'], ascending=[False, False, True]).to_string())
