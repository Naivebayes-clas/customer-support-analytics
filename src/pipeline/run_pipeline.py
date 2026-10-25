"""
End-to-end pipeline: DuckDB → Parquet → Dashboard CSVs
"""
import duckdb
import pandas as pd
import os

def run():
    os.makedirs("./output/parquet", exist_ok=True)
    os.makedirs("./output/csv", exist_ok=True)

    con = duckdb.connect()

    # Load all tables
    con.execute("CREATE TABLE accounts AS SELECT * FROM read_csv_auto('./data/ravenstack_accounts.csv')")
    con.execute("CREATE TABLE subscriptions AS SELECT * FROM read_csv_auto('./data/ravenstack_subscriptions.csv')")
    con.execute("CREATE TABLE churn_events AS SELECT * FROM read_csv_auto('./data/ravenstack_churn_events.csv')")
    con.execute("CREATE TABLE tickets AS SELECT * FROM read_csv_auto('./data/ravenstack_support_tickets.csv')")
    con.execute("CREATE TABLE usage AS SELECT * FROM read_csv_auto('./data/ravenstack_feature_usage.csv')")

    n = con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    print(f"Loaded {n} accounts into DuckDB.")

    # Export to Parquet
    for table in ["accounts", "subscriptions", "churn_events", "tickets", "usage"]:
        con.execute(f"COPY {table} TO './output/parquet/{table}.parquet' (FORMAT PARQUET)")
    print("Exported 5 Parquet files.")

    # --- SQL Analytics ---

    # 1. CSAT by priority (proxy for A/B: high-priority vs. low-priority handling)
    csat_by_priority = con.execute("""
        SELECT priority,
               COUNT(*) as n_tickets,
               AVG(satisfaction_score) as mean_csat,
               AVG(resolution_time_hours) as mean_resolution_hrs,
               AVG(first_response_time_minutes) as mean_first_response_min
        FROM tickets
        WHERE satisfaction_score IS NOT NULL
        GROUP BY priority
        ORDER BY mean_csat DESC
    """).df()
    csat_by_priority.to_csv("./output/csv/csat_by_priority.csv", index=False)
    print("\nCSAT by Priority:")
    print(csat_by_priority.to_string(index=False))

    # 2. Churn by plan tier
    churn_by_plan = con.execute("""
        SELECT plan_tier,
               COUNT(*) as n_accounts,
               SUM(CASE WHEN churn_flag THEN 1 ELSE 0 END) as n_churned,
               ROUND(AVG(CASE WHEN churn_flag THEN 1.0 ELSE 0.0 END) * 100, 1) as churn_pct
        FROM accounts
        GROUP BY plan_tier
        ORDER BY churn_pct DESC
    """).df()
    churn_by_plan.to_csv("./output/csv/churn_by_plan.csv", index=False)
    print("\nChurn by Plan Tier:")
    print(churn_by_plan.to_string(index=False))

    # 3. Churn reasons
    churn_reasons = con.execute("""
        SELECT reason_code,
               COUNT(*) as n_events,
               AVG(refund_amount_usd) as avg_refund
        FROM churn_events
        GROUP BY reason_code
        ORDER BY n_events DESC
    """).df()
    churn_reasons.to_csv("./output/csv/churn_reasons.csv", index=False)
    print("\nChurn Reasons:")
    print(churn_reasons.to_string(index=False))

    # 4. Ticket volume by industry
    tickets_by_industry = con.execute("""
        SELECT a.industry,
               COUNT(t.ticket_id) as n_tickets,
               AVG(t.resolution_time_hours) as mean_resolution_hrs,
               AVG(t.satisfaction_score) as mean_csat
        FROM tickets t
        JOIN accounts a ON t.account_id = a.account_id
        GROUP BY a.industry
        ORDER BY n_tickets DESC
    """).df()
    tickets_by_industry.to_csv("./output/csv/tickets_by_industry.csv", index=False)
    print("\nTickets by Industry:")
    print(tickets_by_industry.to_string(index=False))

    # 5. Dashboard summary
    dashboard = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM accounts) as total_accounts,
            (SELECT COUNT(*) FROM tickets) as total_tickets,
            (SELECT COUNT(*) FROM churn_events) as total_churn_events,
            (SELECT AVG(satisfaction_score) FROM tickets WHERE satisfaction_score IS NOT NULL) as overall_csat,
            (SELECT AVG(resolution_time_hours) FROM tickets) as avg_resolution_hrs,
            (SELECT AVG(CASE WHEN churn_flag THEN 1.0 ELSE 0.0 END) FROM accounts) as churn_rate
    """).df()
    dashboard.to_csv("./output/csv/dashboard.csv", index=False)
    print("\nDashboard Summary:")
    print(dashboard.to_string(index=False))

    con.close()
    print("\n✅ Pipeline complete. Outputs in ./output/")

if __name__ == "__main__":
    run()   
