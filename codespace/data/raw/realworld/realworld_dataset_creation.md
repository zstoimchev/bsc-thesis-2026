# Real-world Dataset Creation

Generated (UTC): 2026-04-21T15:22:38.050613Z

## Source PCAP captures

| File | Packets | Flows | First packet (UTC) | Last packet (UTC) |
|---|---:|---:|---|---|
| out_14042026.pcap | 9,863,195 | 1,121,088 | 2026-04-06T14:44:15.291882Z | 2026-04-15T08:35:42.377298Z |
| out_17042026.pcap | 854,057 | 165,944 | 2026-04-17T15:43:51.386909Z | 2026-04-18T08:45:49.794323Z |
| out_18042026.pcap | 442 | 17 | 2026-04-18T08:50:16.260897Z | 2026-04-18T08:51:35.949058Z |

## Generated datasets

- CIC-like parquet: `C:\Users\dnllz\Desktop\Random\Thesis Paper\models\parquets\realworld\realworld_cic_like.parquet`
- NSL-like parquet: `C:\Users\dnllz\Desktop\Random\Thesis Paper\models\parquets\realworld\realworld_nsl_like.parquet`

## Labeling method

- This dataset uses **weak heuristic labels** from flow behavior (packet rate, byte rate, TCP SYN activity).
- Target malicious ratio: 5.00%
- Final label distribution: {'benign': 1222342, 'malicious_heuristic': 64707}

## Capture context (from server capture filter)

- Capture rule focused on TCP SYN/ACK traffic on selected service ports.
- Configured ports: 8101, 80, 443, 25, 22, 9292, 5000, 25565, 25566, 3389, 8080, 8081, 23, 587, 465, 993, 995, 53, 67, 68, 123, 3306, 5432, 1433, 445
- Services include SMTP, SSH (including custom 8081), web apps, DNS, databases, RDP, SMB, and game server ports.

## Notes

- Labels are not ground truth. Treat as pseudo-labels for model stress-testing.
- Use this dataset primarily for comparative and transfer evaluation, not absolute attack attribution.