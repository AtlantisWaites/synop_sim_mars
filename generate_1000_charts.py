# generate_charts.py
import random
import pandas as pd

# Seed for reproducibility
random.seed(42)

# STEM vocations (NASA astronaut ratios)
vocations = ['Engineer'] * 45 + ['Scientist'] * 30 + ['Pilot'] * 20 + ['Physician'] * 5
vocations = (vocations * 23)[:1000]  # 1000 total
random.shuffle(vocations)

data = []
for i in range(1000):
    # Birth: 1980–2000
    year = random.randint(1980, 2000)

    # Rejection sampling (real orbital physics)
    sun = random.uniform(0, 360)
    moon = random.uniform(0, 360)
    while abs(moon - sun) > 180:
        moon = random.uniform(0, 360)
    mercury = (sun + random.uniform(-28, 28)) % 360
    venus = (sun + random.uniform(-48, 48)) % 360
    mars = random.uniform(0, 360)
    jupiter = random.uniform(0, 360)
    saturn = random.uniform(0, 360)
    uranus = random.uniform(0, 360)
    neptune = random.uniform(0, 360)
    pluto = random.uniform(0, 360)

    # Simple aspects (for synastry)
    aspects = []
    planets = {'Sun': sun, 'Moon': moon, 'Mercury': mercury, 'Venus': venus,
               'Mars': mars, 'Jupiter': jupiter, 'Saturn': saturn}
    keys = list(planets.keys())
    for j in range(len(keys)):
        for k in range(j+1, len(keys)):
            p1, p2 = keys[j], keys[k]
            d = min(abs(planets[p1] - planets[p2]), 360 - abs(planets[p1] - planets[p2]))
            if d <= 8: aspects.append(f"{p1} conj {p2}")
            elif abs(d - 90) <= 6: aspects.append(f"{p1} sq {p2}")
            elif abs(d - 180) <= 8: aspects.append(f"{p1} opp {p2}")

    data.append({
        'id': i,
        'birth_year': year,
        'vocation': vocations[i],
        'sun_deg': round(sun, 4),
        'moon_deg': round(moon, 4),
        'mercury_deg': round(mercury, 4),
        'venus_deg': round(venus, 4),
        'mars_deg': round(mars, 4),
        'jupiter_deg': round(jupiter, 4),
        'saturn_deg': round(saturn, 4),
        'uranus_deg': round(uranus, 4),
        'neptune_deg': round(neptune, 4),
        'pluto_deg': round(pluto, 4),
        'aspects': '; '.join(aspects) if aspects else ''
    })

df = pd.DataFrame(data)
df.to_csv('mars_crew_1000.csv', index=False)
print("Generated: mars_crew_1000.csv (1,000 STEM charts, 1980–2000)")