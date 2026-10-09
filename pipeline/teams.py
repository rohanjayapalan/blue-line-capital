"""
Facts about the 33 franchises we ever see in the data (32 today + Arizona).
Used for travel distance, time zones, and matching names across websites.
"""
import math

# abbrev: (city/place, nickname, arena latitude, arena longitude, UTC offset in winter)
TEAMS = {
    "ANA": ("Anaheim", "Ducks", 33.8078, -117.8765, -8),
    "ARI": ("Arizona", "Coyotes", 33.5319, -112.2611, -7),
    "BOS": ("Boston", "Bruins", 42.3662, -71.0621, -5),
    "BUF": ("Buffalo", "Sabres", 42.8750, -78.8764, -5),
    "CGY": ("Calgary", "Flames", 51.0374, -114.0519, -7),
    "CAR": ("Carolina", "Hurricanes", 35.8033, -78.7219, -5),
    "CHI": ("Chicago", "Blackhawks", 41.8807, -87.6742, -6),
    "COL": ("Colorado", "Avalanche", 39.7487, -105.0077, -7),
    "CBJ": ("Columbus", "Blue Jackets", 39.9693, -83.0061, -5),
    "DAL": ("Dallas", "Stars", 32.7905, -96.8103, -6),
    "DET": ("Detroit", "Red Wings", 42.3411, -83.0553, -5),
    "EDM": ("Edmonton", "Oilers", 53.5469, -113.4979, -7),
    "FLA": ("Florida", "Panthers", 26.1584, -80.3256, -5),
    "LAK": ("Los Angeles", "Kings", 34.0430, -118.2673, -8),
    "MIN": ("Minnesota", "Wild", 44.9448, -93.1010, -6),
    "MTL": ("Montréal", "Canadiens", 45.4961, -73.5693, -5),
    "NSH": ("Nashville", "Predators", 36.1592, -86.7785, -6),
    "NJD": ("New Jersey", "Devils", 40.7334, -74.1711, -5),
    "NYI": ("New York", "Islanders", 40.7117, -73.7255, -5),
    "NYR": ("New York", "Rangers", 40.7505, -73.9934, -5),
    "OTT": ("Ottawa", "Senators", 45.2969, -75.9272, -5),
    "PHI": ("Philadelphia", "Flyers", 39.9012, -75.1720, -5),
    "PIT": ("Pittsburgh", "Penguins", 40.4394, -79.9892, -5),
    "SJS": ("San Jose", "Sharks", 37.3327, -121.9010, -8),
    "SEA": ("Seattle", "Kraken", 47.6221, -122.3540, -8),
    "STL": ("St. Louis", "Blues", 38.6268, -90.2026, -6),
    "TBL": ("Tampa Bay", "Lightning", 27.9427, -82.4518, -5),
    "TOR": ("Toronto", "Maple Leafs", 43.6435, -79.3791, -5),
    "UTA": ("Utah", "Mammoth", 40.7683, -111.9011, -7),
    "VAN": ("Vancouver", "Canucks", 49.2778, -123.1089, -8),
    "VGK": ("Vegas", "Golden Knights", 36.1029, -115.1784, -8),
    "WSH": ("Washington", "Capitals", 38.8981, -77.0209, -5),
    "WPG": ("Winnipeg", "Jets", 49.8928, -97.1436, -6),
}

# NHL's numeric team ids (read from real game files)
NHL_IDS = {"ANA": 24, "ARI": 53, "BOS": 6, "BUF": 7, "CAR": 12, "CBJ": 29, "CGY": 20, "CHI": 16, "COL": 21, "DAL": 25, "DET": 17, "EDM": 22, "FLA": 13, "LAK": 26, "MIN": 30, "MTL": 8, "NJD": 1, "NSH": 18, "NYI": 2, "NYR": 3, "OTT": 9, "PHI": 4, "PIT": 5, "SEA": 55, "SJS": 28, "STL": 19, "TBL": 14, "TOR": 10, "UTA": 68, "VAN": 23, "VGK": 54, "WPG": 52, "WSH": 15}
ID_TO_ABBREV = {v: k for k, v in NHL_IDS.items()}

# When a franchise moves, its rating carries over to the new team.
RELOCATIONS = {"ARI": "UTA"}          # Arizona Coyotes -> Utah (2024-25)
EXPANSION = {"VGK": 20172018, "SEA": 20212022}

# Daily Faceoff uses long slugs in its URLs.
DFO_SLUGS = {
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres",
    "CGY": "calgary-flames", "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche", "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars",
    "DET": "detroit-red-wings", "EDM": "edmonton-oilers", "FLA": "florida-panthers",
    "LAK": "los-angeles-kings", "MIN": "minnesota-wild", "MTL": "montreal-canadiens",
    "NSH": "nashville-predators", "NJD": "new-jersey-devils", "NYI": "new-york-islanders",
    "NYR": "new-york-rangers", "OTT": "ottawa-senators", "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks", "SEA": "seattle-kraken",
    "STL": "st-louis-blues", "TBL": "tampa-bay-lightning", "TOR": "toronto-maple-leafs",
    "UTA": "utah-mammoth", "VAN": "vancouver-canucks", "VGK": "vegas-golden-knights",
    "WSH": "washington-capitals", "WPG": "winnipeg-jets",
}
SLUG_TO_ABBREV = {slug: ab for ab, slug in DFO_SLUGS.items()}
# older slug Daily Faceoff used before Utah picked "Mammoth"
SLUG_TO_ABBREV["utah-hockey-club"] = "UTA"

# ESPN writes some teams differently.
ESPN_ALIASES = {"NJ": "NJD", "SJ": "SJS", "TB": "TBL", "LA": "LAK", "UTAH": "UTA",
                "VEG": "VGK", "LV": "VGK", "WAS": "WSH", "MON": "MTL", "CLB": "CBJ",
                "NAS": "NSH", "PHX": "ARI"}

# The Odds API uses full names.
FULL_NAMES = {ab: f"{t[0]} {t[1]}" for ab, t in TEAMS.items()}
FULL_NAMES["MTL"] = "Montreal Canadiens"
FULL_NAMES["UTA"] = "Utah Mammoth"
NAME_TO_ABBREV = {v.lower(): k for k, v in FULL_NAMES.items()}
NAME_TO_ABBREV["utah hockey club"] = "UTA"
NAME_TO_ABBREV["st louis blues"] = "STL"
NAME_TO_ABBREV["montréal canadiens"] = "MTL"


def current_franchise(abbrev):
    """ARI -> UTA, everyone else unchanged."""
    return RELOCATIONS.get(abbrev, abbrev)


def distance_km(team_a, team_b):
    """Great-circle distance between two arenas (the haversine formula)."""
    if team_a == team_b:
        return 0.0
    _, _, lat1, lon1, _ = TEAMS[team_a]
    _, _, lat2, lon2, _ = TEAMS[team_b]
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def tz_offset(team):
    return TEAMS[team][4]


def display_name(team):
    place, nick = TEAMS[team][0], TEAMS[team][1]
    return f"{place} {nick}"
