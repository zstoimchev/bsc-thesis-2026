## Short info

- `realworld` folder: dataset files with assumption of 5% malicious (not accurate, just picked at random packets to label as malicious)
- `realworld_suricata` folder: used Suricata IDS rules to identify the actual malicious traffic (16 flows, instead of ~60k labeled in the first dataset; meaning ~0.00124%  malicious)
- `realworld_botnet_balanced` folder: real world data from the 3 captures (look at .md file in realworld folder), and injected ~400,000 flows of malicious data (botnet attack from 80 machines), making the dataset have ~21.6% malicious packets

## Final split:
- 1,287,049 benign
- 400,016 malicious flows

#### Special thanks to [dnllzc](https://github.com/dnllzc) for preparing the super-set...
