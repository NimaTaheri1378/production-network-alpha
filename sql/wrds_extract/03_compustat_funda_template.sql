-- Template only. Annual fundamentals extraction.
select *
from comp_na_daily_all.funda
where datadate between '2015-01-01' and '2025-12-31'
  and indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C';
