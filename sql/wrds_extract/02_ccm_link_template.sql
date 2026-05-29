-- Template only. Point-in-time CRSP/Compustat link extraction.
select *
from crsp_a_ccm.ccmxpf_linktable
where coalesce(linkenddt, date '2099-12-31') >= date '2015-01-01'
  and linkdt <= date '2025-12-31';
