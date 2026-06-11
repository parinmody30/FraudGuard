"""Static reference data for India-centric synthetic generation.

Kept as plain module-level constants (no I/O, no global RNG) so generators can sample
from them deterministically using an injected ``numpy.random.Generator``.
"""

from __future__ import annotations

HOME_COUNTRY = "India"
HOME_CURRENCY = "INR"

# (city, state) pairs covering tier-1/2 Indian cities for geographic diversity.
INDIA_CITIES: tuple[tuple[str, str], ...] = (
    ("Mumbai", "Maharashtra"),
    ("Pune", "Maharashtra"),
    ("Delhi", "Delhi"),
    ("Bengaluru", "Karnataka"),
    ("Mysuru", "Karnataka"),
    ("Chennai", "Tamil Nadu"),
    ("Coimbatore", "Tamil Nadu"),
    ("Hyderabad", "Telangana"),
    ("Kolkata", "West Bengal"),
    ("Ahmedabad", "Gujarat"),
    ("Surat", "Gujarat"),
    ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh"),
    ("Kanpur", "Uttar Pradesh"),
    ("Chandigarh", "Chandigarh"),
    ("Indore", "Madhya Pradesh"),
    ("Bhopal", "Madhya Pradesh"),
    ("Kochi", "Kerala"),
    ("Thiruvananthapuram", "Kerala"),
    ("Visakhapatnam", "Andhra Pradesh"),
    ("Patna", "Bihar"),
    ("Guwahati", "Assam"),
    ("Nagpur", "Maharashtra"),
    ("Vadodara", "Gujarat"),
)

# Foreign (city, country) pairs used for travel and geo-anomaly fraud.
FOREIGN_CITIES: tuple[tuple[str, str], ...] = (
    ("Dubai", "United Arab Emirates"),
    ("Singapore", "Singapore"),
    ("London", "United Kingdom"),
    ("New York", "United States"),
    ("Toronto", "Canada"),
    ("Sydney", "Australia"),
    ("Bangkok", "Thailand"),
    ("Kuala Lumpur", "Malaysia"),
    ("Frankfurt", "Germany"),
    ("Hong Kong", "Hong Kong"),
)

# Occupation -> income band weighting. Income band influences salary range and spend.
OCCUPATIONS: tuple[str, ...] = (
    "software_engineer",
    "teacher",
    "doctor",
    "nurse",
    "shop_owner",
    "farmer",
    "accountant",
    "sales_executive",
    "civil_servant",
    "driver",
    "student",
    "retired",
    "freelancer",
    "homemaker",
    "factory_worker",
)

INCOME_BANDS: tuple[str, ...] = ("low", "lower_mid", "mid", "upper_mid", "high")

# Monthly salary range (INR) per income band.
INCOME_BAND_SALARY: dict[str, tuple[int, int]] = {
    "low": (12_000, 25_000),
    "lower_mid": (25_000, 45_000),
    "mid": (45_000, 90_000),
    "upper_mid": (90_000, 180_000),
    "high": (180_000, 500_000),
}

# Per-occupation distribution over income bands (weights, need not sum to 1).
OCCUPATION_INCOME_WEIGHTS: dict[str, dict[str, float]] = {
    "software_engineer": {"mid": 3, "upper_mid": 4, "high": 2},
    "teacher": {"low": 1, "lower_mid": 3, "mid": 2},
    "doctor": {"upper_mid": 3, "high": 5},
    "nurse": {"lower_mid": 3, "mid": 3},
    "shop_owner": {"lower_mid": 2, "mid": 3, "upper_mid": 2},
    "farmer": {"low": 4, "lower_mid": 3, "mid": 1},
    "accountant": {"mid": 4, "upper_mid": 2},
    "sales_executive": {"lower_mid": 2, "mid": 3, "upper_mid": 1},
    "civil_servant": {"mid": 3, "upper_mid": 3},
    "driver": {"low": 3, "lower_mid": 3},
    "student": {"low": 5, "lower_mid": 1},
    "retired": {"low": 2, "lower_mid": 3, "mid": 2},
    "freelancer": {"lower_mid": 2, "mid": 3, "upper_mid": 2},
    "homemaker": {"low": 3, "lower_mid": 2},
    "factory_worker": {"low": 3, "lower_mid": 3},
}

# Merchant categories with (base_risk, typical_amount_multiplier). base_risk in 0..1.
# The multiplier scales a user's baseline spend for that category.
MERCHANT_CATEGORIES: dict[str, tuple[float, float]] = {
    "groceries": (0.05, 0.6),
    "fuel": (0.07, 0.7),
    "dining": (0.12, 0.5),
    "utilities": (0.05, 0.8),
    "telecom": (0.06, 0.4),
    "ecommerce": (0.20, 0.9),
    "electronics": (0.25, 2.5),
    "apparel": (0.15, 0.8),
    "travel": (0.30, 3.0),
    "entertainment": (0.18, 0.5),
    "healthcare": (0.10, 1.2),
    "education": (0.08, 1.5),
    "jewelry": (0.45, 4.0),
    "gaming": (0.55, 0.6),
    "crypto": (0.85, 3.5),
    "gambling": (0.80, 1.5),
    "money_transfer": (0.50, 2.0),
    "wallet_topup": (0.30, 0.5),
}

# Names used to synthesize merchant brand strings per category.
MERCHANT_NAME_PREFIX: dict[str, tuple[str, ...]] = {
    "groceries": ("BigBasket", "DMart", "Reliance Fresh", "More", "Spencer's"),
    "fuel": ("IndianOil", "HP Petrol", "BharatGas Fuel", "Shell", "Nayara"),
    "dining": ("Swiggy", "Zomato", "Barbeque Nation", "Cafe Coffee Day", "Haldiram's"),
    "utilities": ("Tata Power", "Adani Electricity", "BESCOM", "MSEB", "BWSSB"),
    "telecom": ("Airtel", "Jio", "Vi", "BSNL"),
    "ecommerce": ("Amazon IN", "Flipkart", "Myntra", "Meesho", "Ajio"),
    "electronics": ("Croma", "Reliance Digital", "Vijay Sales", "Apple Store"),
    "apparel": ("Lifestyle", "Pantaloons", "Westside", "Max Fashion"),
    "travel": ("MakeMyTrip", "IRCTC", "IndiGo", "Goibibo", "Ola Cabs"),
    "entertainment": ("PVR Cinemas", "BookMyShow", "Netflix IN", "Spotify"),
    "healthcare": ("Apollo Pharmacy", "1mg", "PharmEasy", "Practo"),
    "education": ("BYJU'S", "Unacademy", "Coursera", "UpGrad"),
    "jewelry": ("Tanishq", "Kalyan Jewellers", "Malabar Gold", "PC Jeweller"),
    "gaming": ("Steam", "Google Play Games", "Dream11", "MPL"),
    "crypto": ("WazirX", "CoinDCX", "ZebPay", "Binance"),
    "gambling": ("BetWay", "Casino Royale", "JungleeRummy", "PokerBaazi"),
    "money_transfer": ("Western Union", "InstantRemit", "QuickCash Transfer"),
    "wallet_topup": ("Paytm Wallet", "PhonePe Wallet", "Amazon Pay", "Mobikwik"),
}

# Device type -> plausible (operating_system, browser) options.
DEVICE_TYPES: tuple[str, ...] = ("android_phone", "iphone", "windows_laptop", "mac_laptop", "android_tablet")

DEVICE_OS: dict[str, tuple[str, ...]] = {
    "android_phone": ("Android 12", "Android 13", "Android 14"),
    "iphone": ("iOS 16", "iOS 17", "iOS 18"),
    "windows_laptop": ("Windows 10", "Windows 11"),
    "mac_laptop": ("macOS Ventura", "macOS Sonoma"),
    "android_tablet": ("Android 12", "Android 13"),
}

DEVICE_BROWSER: dict[str, tuple[str, ...]] = {
    "android_phone": ("Chrome Mobile", "Samsung Internet", "App/Android"),
    "iphone": ("Safari Mobile", "Chrome iOS", "App/iOS"),
    "windows_laptop": ("Chrome", "Edge", "Firefox"),
    "mac_laptop": ("Safari", "Chrome", "Firefox"),
    "android_tablet": ("Chrome Mobile", "App/Android"),
}

# First-name / last-name pools for beneficiary names.
FIRST_NAMES: tuple[str, ...] = (
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna",
    "Ishaan", "Ananya", "Diya", "Aadhya", "Saanvi", "Pari", "Anika", "Navya",
    "Riya", "Myra", "Kiara", "Aarohi", "Rahul", "Priya", "Neha", "Rohan",
)
LAST_NAMES: tuple[str, ...] = (
    "Sharma", "Verma", "Gupta", "Iyer", "Reddy", "Nair", "Patel", "Singh",
    "Das", "Mukherjee", "Rao", "Joshi", "Kulkarni", "Menon", "Pillai", "Chauhan",
)

# Active-hour archetypes (hours of day, 0..23) -> diversity in time-of-day behavior.
ACTIVE_HOUR_ARCHETYPES: dict[str, tuple[int, ...]] = {
    "early_bird": (6, 7, 8, 9, 10, 11, 12, 18, 19, 20),
    "nine_to_five": (8, 9, 12, 13, 17, 18, 19, 20, 21),
    "night_owl": (10, 11, 12, 19, 20, 21, 22, 23, 0, 1),
    "all_day": (7, 9, 11, 13, 15, 17, 19, 21, 22),
}

# Monthly seasonality multipliers (index 0 = January). Festival season Oct-Nov peaks.
MONTH_SEASONALITY: tuple[float, ...] = (
    0.95, 0.90, 1.00, 1.00, 1.05, 0.95,
    1.00, 1.05, 1.10, 1.30, 1.35, 1.15,
)
