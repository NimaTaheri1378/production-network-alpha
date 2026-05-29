-- Template only. RavenPack equities annual partition union.
-- Add column projection and date filters after Phase 1 confirms exact timestamp/date columns.
select * from ravenpack_dj.rpa_djpr_equities_2015
union all
select * from ravenpack_dj.rpa_djpr_equities_2016
union all
select * from ravenpack_dj.rpa_djpr_equities_2017
union all
select * from ravenpack_dj.rpa_djpr_equities_2018
union all
select * from ravenpack_dj.rpa_djpr_equities_2019
union all
select * from ravenpack_dj.rpa_djpr_equities_2020
union all
select * from ravenpack_dj.rpa_djpr_equities_2021
union all
select * from ravenpack_dj.rpa_djpr_equities_2022
union all
select * from ravenpack_dj.rpa_djpr_equities_2023
union all
select * from ravenpack_dj.rpa_djpr_equities_2024
union all
select * from ravenpack_dj.rpa_djpr_equities_2025;
