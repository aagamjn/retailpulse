-- RetailPulse: MySQL Schema
-- Run this in MySQL Workbench or CLI before loading data

CREATE DATABASE IF NOT EXISTS retailpulse;
USE retailpulse;

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INT PRIMARY KEY,
    country       VARCHAR(100)
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    stock_code    VARCHAR(20) PRIMARY KEY,
    description   VARCHAR(255)
);

-- Invoices table
CREATE TABLE IF NOT EXISTS invoices (
    invoice_no    VARCHAR(20)  PRIMARY KEY,
    invoice_date  DATETIME     NOT NULL,
    customer_id   INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Invoice line items
CREATE TABLE IF NOT EXISTS invoice_items (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    invoice_no    VARCHAR(20)  NOT NULL,
    stock_code    VARCHAR(20)  NOT NULL,
    quantity      INT          NOT NULL,
    unit_price    DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (invoice_no)  REFERENCES invoices(invoice_no),
    FOREIGN KEY (stock_code)  REFERENCES products(stock_code)
);

-- Useful view: full flattened transactions
CREATE OR REPLACE VIEW vw_transactions AS
SELECT
    ii.id,
    i.invoice_no,
    i.invoice_date,
    i.customer_id,
    c.country,
    ii.stock_code,
    p.description,
    ii.quantity,
    ii.unit_price,
    ROUND(ii.quantity * ii.unit_price, 2) AS revenue
FROM invoice_items ii
JOIN invoices  i ON ii.invoice_no  = i.invoice_no
JOIN customers c ON i.customer_id  = c.customer_id
JOIN products  p ON ii.stock_code  = p.stock_code;