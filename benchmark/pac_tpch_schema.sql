-- TPC-H PAC Links Configuration
-- This file adds PAC LINK metadata to existing TPC-H tables
-- Execute this after the tables are created and loaded with data
-- PAC LINKs define foreign key relationships for PAC compilation

-- Protected columns in customer table
ALTER PAC TABLE customer ADD PROTECTED (c_custkey);
ALTER PAC TABLE customer ADD PROTECTED (c_comment);
ALTER PAC TABLE customer ADD PROTECTED (c_acctbal);
ALTER PAC TABLE customer ADD PROTECTED (c_name);
ALTER PAC TABLE customer ADD PROTECTED (c_address);

-- Orders -> Customer link
ALTER PAC TABLE orders ADD PAC LINK (o_custkey) REFERENCES customer(c_custkey);

-- Lineitem -> Orders link
ALTER PAC TABLE lineitem ADD PAC LINK (l_orderkey) REFERENCES orders(o_orderkey);
