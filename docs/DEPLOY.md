# Putting TwinSim on your own domain

TwinSim is not a static site. The browser talks to a Python server that runs the language model, so you
need a machine that can run Python around the clock. A small VPS (1 vCPU, 2 GB RAM) is enough to serve
the trained `dev` model; training new versions is slow on such a machine but works.

The steps below use a Linux VPS, Docker and Caddy (which gets HTTPS certificates automatically).
Replace `twinsim.example.com` with your own domain everywhere.

## 1. Buy a domain and a server
* Domain: any registrar (Namecheap, Cloudflare, GoDaddy, Hostinger, ...).
* Server: any VPS with Ubuntu 22.04 or 24.04 (DigitalOcean, Hetzner, Linode, AWS Lightsail, ...).
  Note its public IPv4 address.

## 2. Point the domain at the server
In your registrar's DNS settings add one record:

| Type | Name | Value |
|---|---|---|
| A | `twinsim` (or `@` for the bare domain) | your server's IPv4 address |

DNS changes can take from a few minutes to a few hours. Check with `nslookup twinsim.example.com`.

## 3. Build the frontend (on your computer)
```bash
npm install
npm run build            # creates dist/
```
The zip you received already contains a built `dist/`, so you can skip this step unless you changed `src/`.

## 4. Copy the project to the server
```bash
scp -r twinsim user@SERVER_IP:~/
```

## 5. Run TwinSim in Docker (on the server)
```bash
sudo apt update && sudo apt install -y docker.io
cd ~/twinsim
sudo docker build -t twinsim .
sudo mkdir -p /srv/twinsim && sudo cp -r runtime/* /srv/twinsim/     # trained models + database
sudo docker run -d --name twinsim --restart unless-stopped \
  -p 127.0.0.1:8000:8000 -v /srv/twinsim:/data twinsim
curl http://127.0.0.1:8000/api/health                                # should print {"ok":true}
```
The container starts in public mode (`TWINSIM_PUBLIC=1`). Visitors can build twins, run scenarios
and rate answers, but cannot start training, change settings or switch the active model. Run those yourself
on the server:
```bash
sudo docker exec -it twinsim python scripts/train.py info
sudo docker exec -it twinsim python scripts/train.py learn --iterations 3
```

## 6. HTTPS with Caddy
```bash
sudo apt install -y caddy
sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
twinsim.example.com {
    reverse_proxy 127.0.0.1:8000
}
EOF
sudo systemctl reload caddy
```
Caddy requests a Let's Encrypt certificate on first visit. Open `https://twinsim.example.com`.

## 7. Before you share the link
* Fill in your name, contact email and domain in `src/site.js`, then rebuild (`npm run build`) and redeploy.
  These appear on the Privacy and Terms pages.
* Read the Privacy and Terms pages and adjust them to your situation. They describe what the code actually
  does, but they are a starting point, not legal advice.
* Back up `/srv/twinsim` (it holds the models and every experiment).

## Updating
```bash
cd ~/twinsim && sudo docker build -t twinsim . && sudo docker rm -f twinsim
sudo docker run -d --name twinsim --restart unless-stopped -p 127.0.0.1:8000:8000 -v /srv/twinsim:/data twinsim
```
