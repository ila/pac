SELECT pac_sum(hash(l_orderkey), l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem JOIN part ON lineitem.l_partkey = part.p_partkey
WHERE part.p_brand = 'Brand#23' AND part.p_container = 'MED BOX' AND 
      pac_filter_lt(
        lineitem.l_quantity*5, 
        (SELECT pac_avg_counters(hash(l_sub.rowid), l_sub.l_quantity) 
          FROM lineitem AS l_sub
         WHERE l_sub.l_partkey = part.p_partkey));
