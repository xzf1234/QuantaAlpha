# A-share structured data drop zone

Put CSV, Parquet, or Feather exports from a-stock-data or similar providers in
this directory. QuantaAlpha will normalize and join them to the daily panel
during custom factor calculation.

Required columns:

- `date`, `datetime`, `trade_date`, `ann_date`, or `report_date`
- `instrument`, `symbol`, `code`, `ts_code`, `stock_code`, or `sec_code`

Supported numeric fields include:

- Size/valuation: `market_cap`, `total_mv`, `circ_mv`, `pe`, `pe_ttm`, `pb`,
  `ps`, `turnover_rate`
- Fund flow: `main_net_inflow`, `super_net_inflow`, `large_net_inflow`,
  `north_net_inflow`
- Margin finance: `margin_balance`, `margin_buy`, `short_balance`,
  `short_sell`
- Unlocks: `unlock_amount`, `unlock_ratio`
- Fundamentals: `roe`, `roa`, `gross_margin`, `net_profit_yoy`,
  `revenue_yoy`, `debt_to_asset`

Instrument examples are normalized automatically: `600000.SH`, `SH600000`,
`sh600000`, and `600000` all map to Qlib-style `SH600000`.
