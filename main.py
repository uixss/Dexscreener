#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import yaml
import signal
import logging
import unicodedata
import datetime
from datetime import timezone
from typing import Any, Dict, List, Optional, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import schedule
import cloudscraper

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Index
from sqlalchemy.orm import declarative_base, sessionmaker

LOG_FILE = os.environ.get("DEX_BOT_LOG", "dexscreener_bot.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()])

def safe_log_message(message: Any) -> str:
    return unicodedata.normalize("NFKD", str(message)).encode("ascii", "ignore").decode("ascii")

def signal_handler(signum, frame):
    logging.info("🛑 Señal recibida. Cerrando bot de forma segura…")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def _deep_get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def load_config(config_path: str = None) -> Dict:
    config_path = config_path or os.environ.get("DEX_BOT_CONFIG", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    defaults = {
        'api_urls': {'tokens': '', 'boosts': '', 'boosts_top': ''},
        'coin_blacklist': [],
        'dev_blacklist': [],
        'rugcheck': {'base_url': 'https://api.rugcheck.xyz/v1', 'api_key': '', 'good_status': 'Good'},
        'supply_check': {'bundled_supply_field': 'is_supply_bundled'},
        'fake_volume_detection': {
            'method': 'algorithm',
            'algorithm': {'min_volume_threshold': 10_000.0, 'max_volume_change_percentage': 300.0},
            'pocket_universe': {'base_url': '', 'api_key': ''}
        },
        'filters': {'min_price_change_percentage_24h': 50.0, 'max_price_change_percentage_24h': -50.0, 'monitored_events': ['pumped', 'rugged', 'tier-1', 'listed_on_cex']},
        'database': {'url': 'sqlite:///coins.db'},
        'runtime': {'json_interval_seconds': 180, 'data_dir': 'data'}
    }
    def deep_merge(a: Dict, b: Dict) -> Dict:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    cfg = deep_merge(defaults, raw)
    for key in ['tokens', 'boosts', 'boosts_top']:
        val = cfg['api_urls'].get(key, '').strip()
        if not val:
            logging.warning(f"api_urls.{key} está vacío. Algunas funciones pueden no operar.")
        cfg['api_urls'][key] = val
    logging.info("Configuration loaded successfully.")
    return cfg

CONFIG = load_config()
try:
    _log_level = str(CONFIG.get('logging', {}).get('level', 'INFO')).upper()
    logging.getLogger().setLevel(_log_level)
except Exception:
    pass

Base = declarative_base()

class CoinEvent(Base):
    __tablename__ = 'coin_events'
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String, index=True, nullable=False)
    name = Column(String)
    price = Column(Float)
    event_type = Column(String, index=True)
    dev_address = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.now(timezone.utc), index=True)
    metadata_json = Column(Text)
    __table_args__ = (Index('ix_token_time', 'token_id', 'timestamp'),)

ENGINE = create_engine(CONFIG['database']['url'])
Base.metadata.create_all(ENGINE)
Session = sessionmaker(bind=ENGINE)

DATA_DIR = CONFIG['runtime']['data_dir']
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_FILE = os.path.join(DATA_DIR, "events.json")

DEX_API_URLS = {"tokens": CONFIG['api_urls']['tokens'], "boosts": CONFIG['api_urls']['boosts'], "boosts_top": CONFIG['api_urls']['boosts_top']}
TELEGRAM_CFG = CONFIG.get('telegram', {})

def make_retrying_session(total=3, backoff=0.5, status_forcelist=(429, 500, 502, 503, 504)) -> requests.Session:
    sess = requests.Session()
    retry = Retry(total=total, read=total, connect=total, backoff_factor=backoff, status_forcelist=status_forcelist, allowed_methods=("GET", "POST"))
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount('http://', adapter)
    sess.mount('https://', adapter)
    return sess

REQ = make_retrying_session()
SCRAPER = cloudscraper.create_scraper()

def save_to_file(data: Any, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True
    )
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)
    logging.info(f"Datos guardados en {filename}")

def load_from_file(filename: str) -> Any:
    if not os.path.exists(filename):
        logging.warning(f"Archivo no encontrado: {filename}. Iniciando con lista vacía.")
        return []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error al leer {filename}: {e}")
        return []

def append_event_json(event: Dict) -> None:
    events = load_from_file(DATABASE_FILE)
    if not isinstance(events, list):
        events = []
    events.append(event)
    save_to_file(events, DATABASE_FILE)

def fetch_api_data(url: str, use_cloudscraper: bool = True) -> Optional[Dict]:
    if not url:
        return None
    try:
        sess = SCRAPER if use_cloudscraper else REQ
        resp = sess.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Error fetching {url}: {safe_log_message(e)}")
        return None

def update_dex_data() -> None:
    for name, url in DEX_API_URLS.items():
        if not url:
            continue
        logging.info(f"📡 Obteniendo datos de {name}")
        data = fetch_api_data(url)
        if data is None:
            continue
        tokens = data.get("tokens", data) if isinstance(data, dict) else data
        filename = os.path.join(DATA_DIR, f"dex_{name}.json")
        save_to_file(tokens, filename)

def send_telegram_message(text: str) -> None:
    token = (TELEGRAM_CFG.get('bot_token') or '').strip()
    chat_id = (TELEGRAM_CFG.get('chat_id') or '').strip()
    if not token or not chat_id:
        return
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML', 'disable_web_page_preview': True}
    try:
        r = REQ.post(api_url, data=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Telegram send error: {e}")

def is_blacklisted(token_id: str, token_name: str, dev_address: str) -> bool:
    coin_blacklist = [str(s).lower() for s in CONFIG.get('coin_blacklist', [])]
    dev_blacklist = [str(s).lower() for s in CONFIG.get('dev_blacklist', [])]
    return (str(token_id).lower() in coin_blacklist or str(token_name).lower() in coin_blacklist or str(dev_address).lower() in dev_blacklist)

def is_token_good_rugcheck(token_id: str) -> bool:
    base = (CONFIG['rugcheck'].get('base_url') or 'https://api.rugcheck.xyz/v1').rstrip('/')
    api_key = (CONFIG['rugcheck'].get('api_key') or '').strip()
    if not token_id or token_id.startswith('0x'):
        return True
    headers = {}
    if api_key:
        headers['X-API-KEY'] = api_key
    try:
        url = f"{base}/tokens/{token_id}/report/summary"
        r = REQ.get(url, headers=headers, timeout=15)
        if r.status_code == 404:
            return True
        r.raise_for_status()
        j = r.json() or {}
        risk_level = (j.get('risk_level') or j.get('riskLevel') or '').upper()
        status = (j.get('status') or '').upper()
        score = 0.0
        try:
            score = float(j.get('score') or j.get('risk_score') or 0)
        except Exception:
            score = 0.0
        if risk_level in {'LOW', 'GOOD', 'OK'} or status in {'GOOD', 'OK'} or score >= 70:
            return True
        return False
    except Exception:
        try:
            url = f"{base}/tokens/{token_id}/report"
            r = REQ.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            j = r.json() or {}
            risk = j.get('risk') or {}
            risk_level = (risk.get('level') or '').upper()
            status = (j.get('status') or '').upper()
            score = 0.0
            try:
                score = float(risk.get('score') or 0)
            except Exception:
                score = 0.0
            if risk_level in {'LOW', 'GOOD', 'OK'} or status in {'GOOD', 'OK'} or score >= 70:
                return True
            return False
        except Exception:
            return False

def is_supply_bundled(token_data: Dict) -> bool:
    field = CONFIG['supply_check'].get('bundled_supply_field', 'is_supply_bundled')
    val = token_data.get(field, False)
    return bool(val)

def update_blacklists_if_bundled(token_data: Dict) -> None:
    coin_id = token_data.get('id')
    dev_address = token_data.get('developer_address', '')
    coin_name = token_data.get('name', 'Unknown')
    if not coin_id or not dev_address:
        return
    CONFIG.setdefault('coin_blacklist', [])
    CONFIG.setdefault('dev_blacklist', [])
    if coin_id not in CONFIG['coin_blacklist']:
        CONFIG['coin_blacklist'].append(coin_id)
    if coin_name not in CONFIG['coin_blacklist']:
        CONFIG['coin_blacklist'].append(coin_name)
    if dev_address not in CONFIG['dev_blacklist']:
        CONFIG['dev_blacklist'].append(dev_address)

def is_volume_valid_algorithm(coin: Dict) -> bool:
    try:
        vol = float(coin.get('daily_volume', 0) or 0)
        change = float(coin.get('volume_change_percentage_24h', 0) or 0)
        min_vol = float(_deep_get(CONFIG, ['fake_volume_detection', 'algorithm', 'min_volume_threshold'], 10_000.0))
        max_change = float(_deep_get(CONFIG, ['fake_volume_detection', 'algorithm', 'max_volume_change_percentage'], 300.0))
        if vol < min_vol:
            return False
        if abs(change) > max_change:
            return False
        return True
    except Exception:
        return False

def is_volume_valid_pocket_universe(coin: Dict) -> bool:
    try:
        pu = CONFIG['fake_volume_detection'].get('pocket_universe', {})
        url = pu.get('base_url', '').strip()
        api_key = pu.get('api_key', '').strip()
        if not url:
            return False
        headers = {'Authorization': f"Bearer {api_key}"} if api_key else {}
        params = {'coin_id': coin.get('id')}
        response = REQ.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        result = response.json() or {}
        is_fake = bool(result.get('is_volume_fake', False))
        return not is_fake
    except Exception:
        return False

def is_volume_valid(coin: Dict) -> bool:
    method = str(CONFIG.get('fake_volume_detection', {}).get('method', 'algorithm')).lower()
    if method == 'pocket_universe':
        return is_volume_valid_pocket_universe(coin)
    return is_volume_valid_algorithm(coin)

def determine_event_type(price_change: float, item: Dict) -> str:
    filters = CONFIG['filters']
    min_pump = float(filters.get('min_price_change_percentage_24h', 50))
    max_dump = float(filters.get('max_price_change_percentage_24h', -50))
    if price_change >= min_pump:
        return 'pumped'
    if price_change <= max_dump:
        return 'rugged'
    if item.get('is_tier_1', False):
        return 'tier-1'
    if item.get('is_listed_on_cex', False):
        return 'listed_on_cex'
    return 'other'

def fetch_coin_data() -> Optional[Dict]:
    url = CONFIG['api_urls'].get('tokens', '').strip()
    if not url:
        logging.warning("api_urls.tokens vacío. No se puede ejecutar fetch_coin_data().")
        return None
    try:
        response = REQ.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            data = {"tokens": data}
        logging.info("Fetched coin data successfully (SQLAlchemy pipeline).")
        return data
    except requests.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

def parse_coin_data(raw_data: Dict) -> List[Dict]:
    coins: List[Dict] = []
    coin_blacklist: Set[str] = set(str(x) for x in CONFIG.get('coin_blacklist', []))
    dev_blacklist: Set[str] = set(str(x) for x in CONFIG.get('dev_blacklist', []))
    monitored_events: Set[str] = set(CONFIG['filters'].get('monitored_events', []))
    tokens_iter = []
    if isinstance(raw_data, dict):
        tokens_iter = raw_data.get('tokens') or []
    elif isinstance(raw_data, list):
        tokens_iter = raw_data
    for item in tokens_iter:
        try:
            coin_id = item.get('id')
            coin_name = item.get('name', 'Unknown')
            price = float(item.get('price', 0) or 0)
            price_change = float(item.get('price_change_percentage_24h', 0) or 0)
            dev_address = str(item.get('developer_address', '')).lower()
            if not coin_id:
                continue
            if coin_id in coin_blacklist or coin_name in coin_blacklist or dev_address in dev_blacklist:
                continue
            event_type = determine_event_type(price_change, item)
            if event_type not in monitored_events:
                continue
            if not is_token_good_rugcheck(coin_id):
                continue
            if is_supply_bundled(item):
                update_blacklists_if_bundled(item)
                continue
            if not is_volume_valid(item):
                continue
            coin = {
                'token_id': coin_id,
                'name': coin_name,
                'price': price,
                'event_type': event_type,
                'dev_address': dev_address,
                'timestamp': datetime.datetime.now(timezone.utc),
                'metadata_json': json.dumps(item, default=str)
            }
            coins.append(coin)
        except Exception as e:
            logging.error(f"Error parsing coin {item.get('id', 'Unknown')}: {e}")
    logging.info(f"✅ Parsed {len(coins)} valid coins (SQLAlchemy pipeline).")
    return coins

def save_to_database(coins: List[Dict]) -> None:
    if not coins:
        return
    session = Session()
    try:
        for coin in coins:
            exists = session.query(CoinEvent).filter(CoinEvent.token_id == coin['token_id'], CoinEvent.timestamp == coin['timestamp']).first()
            if exists:
                continue
            session.add(CoinEvent(**coin))
            try:
                if coin.get('event_type') in set(CONFIG['filters'].get('monitored_events', [])):
                    msg = (f"<b>{coin['event_type'].upper()}</b>\n"
                           f"<b>{coin['name']}</b>\n"
                           f"Price: ${coin['price']:.6f}\n"
                           f"CA: <code>{coin['token_id']}</code>")
                    send_telegram_message_with_button(msg, coin['token_id'], json.loads(coin['metadata_json']))
            except Exception:
                pass
        session.commit()
        logging.info(f"💾 Saved {len(coins)} events to SQLite database.")
    except Exception as e:
        session.rollback()
        logging.error(f"DB save error: {e}")
    finally:
        session.close()

def analyze_data() -> None:
    session = Session()
    try:
        data = session.query(CoinEvent).all()
        if not data:
            logging.info("No data to analyze.")
            return
        df = pd.DataFrame([{'event_type': e.event_type, 'price': e.price, 'timestamp': e.timestamp} for e in data])
        counts = df['event_type'].value_counts()
        logging.info(f"📊 Event Counts:\n{counts.to_string()}")
        logging.info(f"📈 Price Stats:\n{df['price'].describe()}")
    except Exception as e:
        logging.error(f"Analysis error: {e}")
    finally:
        session.close()

def job_sqlalchemy() -> None:
    logging.info("🔄 Starting SQLAlchemy pipeline (hourly)…")
    raw_data = fetch_coin_data()
    if raw_data:
        coins = parse_coin_data(raw_data)
        save_to_database(coins)
        analyze_data()
    logging.info("✅ SQLAlchemy pipeline completed.")

def _pick(d: Dict, *paths):
    for path in paths:
        cur = d
        ok = True
        for p in path.split('.'):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            return cur
    return None

def _to_float(x):
    try:
        if x in [None, ""]:
            return 0.0
        return float(str(x).replace(",", ""))
    except:
        return 0.0

def fetch_ds_metrics_by_address(address: str) -> Optional[Dict]:
    try:
        r = REQ.get(f"https://api.dexscreener.com/latest/dex/tokens/{address}", timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        pairs = data.get("pairs") or []
        if not pairs:
            return None
        best = sorted(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0)) or float((p.get("volume", {}) or {}).get("h24", 0)), reverse=True)[0]
        price = float(best.get("priceUsd") or 0.0)
        ch24 = float((best.get("priceChange", {}) or {}).get("h24", 0.0))
        vol24 = float((best.get("volume", {}) or {}).get("h24", 0.0))
        return {"price": price, "ch24": ch24, "vol24": vol24, "chainId": best.get("chainId"), "pairAddress": best.get("pairAddress")}
    except Exception as e:
        logging.debug(f"Dexscreener enrich fail for {address}: {safe_log_message(e)}")
        return None

def build_dex_url(item: Dict, contract: str) -> str:
    chain = _pick(item, "chainId", "chain", "chain.id")
    pair = _pick(item, "pairAddress", "pair.address")
    if chain and pair:
        return f"https://dexscreener.com/{chain}/{pair}"
    try:
        r = REQ.get(f"https://api.dexscreener.com/latest/dex/tokens/{contract}", timeout=15)
        r.raise_for_status()
        pairs = (r.json() or {}).get("pairs") or []
        if pairs:
            best = sorted(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0)) or float((p.get("volume", {}) or {}).get("h24", 0)), reverse=True)[0]
            chain = best.get("chainId")
            pair = best.get("pairAddress")
            if chain and pair:
                return f"https://dexscreener.com/{chain}/{pair}"
    except Exception:
        pass
    return f"https://dexscreener.com/search?q={requests.utils.quote(contract)}"

def classify_event_simple(token: Dict) -> str:
    change = float(token.get("price_change_percentage_24h", 0) or 0)
    thresholds = CONFIG['filters']
    min_pump = float(thresholds.get('min_price_change_percentage_24h', 50))
    max_dump = float(thresholds.get('max_price_change_percentage_24h', -50))
    if change >= min_pump:
        return 'pumped'
    if change <= max_dump:
        return 'rugged'
    return 'normal'

def process_tokens() -> int:
    tokens_file = os.path.join(DATA_DIR, "dex_tokens.json")
    dex_tokens_raw = load_from_file(tokens_file)
    processed = 0
    if isinstance(dex_tokens_raw, list) and dex_tokens_raw:
        logging.info(f"Sample keys: {list(dex_tokens_raw[0].keys())[:25]}")
    for item in dex_tokens_raw:
        if not isinstance(item, dict):
            continue
        contract = str(_pick(item, "id", "contract", "address", "tokenAddress", "baseToken.address", "token.address", "pairAddress") or "").lower()
        if not contract:
            continue
        name = _pick(item, "name", "header", "baseToken.name", "token.name") or "Unknown"
        price = _to_float(_pick(item, "price", "priceUsd", "priceUSD"))
        ch24 = _to_float(_pick(item, "price_change_percentage_24h", "priceChange24h", "priceChange.h24"))
        vol24 = _to_float(_pick(item, "daily_volume", "volume24h", "volume.h24"))
        vchg24 = _to_float(_pick(item, "volume_change_percentage_24h", "volumeChange24h", "volumeChange.h24"))
        dev = _pick(item, "developer_address") or "N/A"
        if (price == 0.0 and vol24 == 0.0) or ch24 == 0.0:
            m = fetch_ds_metrics_by_address(contract)
            if m:
                price = m.get("price", price)
                ch24 = m.get("ch24", ch24)
                vol24 = m.get("vol24", vol24)
                item.setdefault("chainId", m.get("chainId"))
                item.setdefault("pairAddress", m.get("pairAddress"))
        token = {"id": contract, "name": name, "price": price, "price_change_percentage_24h": ch24, "daily_volume": vol24, "volume_change_percentage_24h": vchg24, "developer_address": dev}
        if is_blacklisted(contract, name, dev):
            continue
        if not is_token_good_rugcheck(contract):
            continue
        if is_supply_bundled(item):
            update_blacklists_if_bundled(item)
            continue
        if not is_volume_valid(token):
            continue
        event_type = classify_event_simple(token)
        if event_type == "normal":
            continue
        event = {"token": token, "event_type": event_type, "timestamp": datetime.datetime.now(timezone.utc).isoformat()}
        append_event_json(event)
        logging.info(f"✅ {event_type.upper()}: {name} (${price:.6f}) | 24h: {ch24:+.2f}% | Vol: ${vol24:,.0f}")
        if event_type in set(CONFIG['filters'].get('monitored_events', [])):
            try:
                msg = (f"<b>{event_type.upper()}</b>\n"
                       f"<b>{name}</b>\n"
                       f"Price: ${price:.6f}\n"
                       f"24h: {ch24:+.2f}%\n"
                       f"Vol: ${vol24:,.0f}\n"
                       f"CA: <code>{contract}</code>")
                send_telegram_message_with_button(msg, contract, item)
            except Exception:
                pass
        processed += 1
    logging.info(f"📊 Procesados {processed} eventos relevantes (JSON pipeline).")
    return processed

import json as _json

def send_telegram_message_with_button(text: str, contract: str, item: Dict = None) -> None:
    token = (TELEGRAM_CFG.get('bot_token') or '').strip()
    chat_id = (TELEGRAM_CFG.get('chat_id') or '').strip()
    if not token or not chat_id:
        return
    dex_url = build_dex_url(item or {}, contract)
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML', 'disable_web_page_preview': False, 'reply_markup': _json.dumps({"inline_keyboard": [[{"text": "🔎 Ver en Dexscreener", "url": dex_url}]]})}
    try:
        r = REQ.post(api_url, data=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Telegram send (button) error: {e}")

def job_json_pipeline() -> None:
    logging.info("🔄 Starting JSON pipeline (every 3 min)…")
    try:
        update_dex_data()
        process_tokens()
        stats: Dict[str, int] = {"total": 0}
        events = load_from_file(DATABASE_FILE)
        if isinstance(events, list):
            stats["total"] = len(events)
            for e in events:
                etype = (e or {}).get("event_type")
                if etype:
                    stats[etype] = stats.get(etype, 0) + 1
        logging.info(f"📈 Stats: {stats}")
    except Exception as e:
        logging.error(f"Error in JSON pipeline: {e}")

def main() -> None:
    logging.info("🚀 Dexscreener Monitoring Bot Iniciado (Doble Pipeline)")
    schedule.every(1).hours.do(job_sqlalchemy)
    logging.info("📅 Scheduler: Pipeline SQLAlchemy cada 1 hora.")
    json_interval = int(CONFIG['runtime'].get('json_interval_seconds', 180))
    last_run = 0.0
    job_sqlalchemy()
    job_json_pipeline()
    while True:
        try:
            schedule.run_pending()
            now = time.time()
            if now - last_run >= json_interval:
                job_json_pipeline()
                last_run = now
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
