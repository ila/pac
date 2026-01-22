-- TPC-H PAC Links Configuration
-- This file adds PAC LINK metadata to existing TPC-H tables
-- Execute this after the tables are created and loaded with data
-- PAC LINKs define foreign key relationships for PAC compilation

-- Protected columns in customer table
ALTER PAC TABLE customer ADD PROTECTED (c_acctbal);
ALTER PAC TABLE customer ADD PROTECTED (c_name);
ALTER PAC TABLE customer ADD PROTECTED (c_address);
ALTER PAC TABLE customer ADD PROTECTED (c_phone);

-- Nation -> Region link
ALTER PAC TABLE nation ADD PAC LINK (n_regionkey) REFERENCES region(r_regionkey);

-- Supplier -> Nation link
ALTER PAC TABLE supplier ADD PAC LINK (s_nationkey) REFERENCES nation(n_nationkey);

-- Customer -> Nation link
ALTER PAC TABLE customer ADD PAC LINK (c_nationkey) REFERENCES nation(n_nationkey);

-- Partsupp -> Part link
ALTER PAC TABLE partsupp ADD PAC LINK (ps_partkey) REFERENCES part(p_partkey);

-- Partsupp -> Supplier link
ALTER PAC TABLE partsupp ADD PAC LINK (ps_suppkey) REFERENCES supplier(s_suppkey);

-- Orders -> Customer link
ALTER PAC TABLE orders ADD PAC LINK (o_custkey) REFERENCES customer(c_custkey);

-- Lineitem -> Orders link
ALTER PAC TABLE lineitem ADD PAC LINK (l_orderkey) REFERENCES orders(o_orderkey);

-- Lineitem -> Partsupp link (composite foreign key - NOW PROPERLY SUPPORTED!)
ALTER PAC TABLE lineitem ADD PAC LINK (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey);
