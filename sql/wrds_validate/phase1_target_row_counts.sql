-- Phase 1 metadata validation. Safe: row counts only.
-- Generated at 2026-05-28T20:11:01Z
select 'crsp_daily_returns' as alias, 'crsp_a_stock.dsf' as table_name, count(*) as n_rows from crsp_a_stock.dsf limit 1;
select 'crsp_names' as alias, 'crsp_a_stock.stocknames' as table_name, count(*) as n_rows from crsp_a_stock.stocknames limit 1;
select 'crsp_delisting' as alias, 'crsp_a_stock.dsedelist' as table_name, count(*) as n_rows from crsp_a_stock.dsedelist limit 1;
select 'crsp_compustat_link' as alias, 'crsp_a_ccm.ccmxpf_linktable' as table_name, count(*) as n_rows from crsp_a_ccm.ccmxpf_linktable limit 1;
select 'compustat_fundamentals' as alias, 'comp_na_daily_all.funda' as table_name, count(*) as n_rows from comp_na_daily_all.funda limit 1;
select 'compustat_company' as alias, 'comp_na_daily_all.company' as table_name, count(*) as n_rows from comp_na_daily_all.company limit 1;
select 'supply_chain_link' as alias, 'wrdsapps_link_supplychain.seglink' as table_name, count(*) as n_rows from wrdsapps_link_supplychain.seglink limit 1;
select 'supply_chain_segments_fallback' as alias, 'comp_segments_hist_daily.wrds_seg_customer' as table_name, count(*) as n_rows from comp_segments_hist_daily.wrds_seg_customer limit 1;
select 'ravenpack_entity_mapping' as alias, 'ravenpack_common.rpa_entity_mappings' as table_name, count(*) as n_rows from ravenpack_common.rpa_entity_mappings limit 1;
select 'ravenpack_company_mapping' as alias, 'ravenpack_common.rpa_company_mappings' as table_name, count(*) as n_rows from ravenpack_common.rpa_company_mappings limit 1;
select 'ibes_detail' as alias, 'tr_ibes.det_epsus' as table_name, count(*) as n_rows from tr_ibes.det_epsus limit 1;
select 'ibes_summary' as alias, 'tr_ibes.statsum_epsus' as table_name, count(*) as n_rows from tr_ibes.statsum_epsus limit 1;
select 'ibes_id' as alias, 'tr_ibes.id' as table_name, count(*) as n_rows from tr_ibes.id limit 1;
select 'crsp_ibes_link' as alias, 'wrdsapps_link_crsp_ibes.ibcrsphist' as table_name, count(*) as n_rows from wrdsapps_link_crsp_ibes.ibcrsphist limit 1;
select 'liquidity_bbd' as alias, 'contrib_liquidity_taq.bbd' as table_name, count(*) as n_rows from contrib_liquidity_taq.bbd limit 1;
select 'liquidity_ilc' as alias, 'contrib_liquidity_taq.ilc' as table_name, count(*) as n_rows from contrib_liquidity_taq.ilc limit 1;
