import os
import random
import csv

def generate_sample_datasets(base_dir: str):
    train_dir = os.path.join(base_dir, "dataset", "train")
    test_dir = os.path.join(base_dir, "dataset", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(42)

    # Base entity templates (US & India for train; US, India, and France for test)
    train_templates = [
        # US entities
        ("Apex Logistics Inc", "1200 Industrial Pkwy, Suite 400, Chicago, IL 60607", "US"),
        ("Beacon Health Services LLC", "450 Lexington Ave, 12th Floor, New York, NY 10017", "US"),
        ("Cascade Mountain Coffee Corp", "88 Pine St, Seattle, WA 98101", "US"),
        ("Delta Precision Engineering", "3050 Technology Dr, Austin, TX 78759", "US"),
        ("Echo River Bookstore Co", "14 Market St, San Francisco, CA 94105", "US"),
        ("Falcon Security Systems", "2100 Elmwood Ave, Buffalo, NY 14207", "US"),
        ("Grand View Motors", "500 Highway 101, San Jose, CA 95110", "US"),
        ("Horizon Organic Bakery", "77 Broadway, Boston, MA 02116", "US"),
        ("Ironclad Industrial Supply", "8900 Commerce Way, Miami, FL 33178", "US"),
        ("Jupiter Data Solutions", "101 Federal St, Pittsburgh, PA 15212", "US"),
        # India entities
        ("Sharma Electronic Enterprises Pvt Ltd", "Plot 42, Sector 18, Electronic City, Bengaluru 560100", "India"),
        ("Maharaja Sweets and Snacks", "Shop 12, Near SBI ATM, MG Road, Pune 411001", "India"),
        ("Balaji Hardware and Sanitary", "14/B, Opposite Old Bus Stand, Ring Road, Surat 395002", "India"),
        ("Sunrise Pharma Chemists", "Shop 5, Ground Floor, Behind Civil Hospital, Ahmedabad 380016", "India"),
        ("Krishna Textiles and Garments", "204 Cotton Market, Near Gandhi Chowk, Surat 395003", "India"),
        ("Patel Automobile Workshop", "Survey No 88, Near Toll Plaza, NH 48, Vadodara 390010", "India"),
        ("Rajesh Kirana and General Store", "Gali No 3, Vikas Nagar, Uttam Nagar, New Delhi 110059", "India"),
        ("Sai Baba Steel Traders", "Plot 105, Phase 2, GIDC Industrial Estate, Rajkot 360002", "India"),
        ("Om Sai Tour and Travels", "Office 3, 2nd Floor, City Center Mall, Jaipur 302001", "India"),
        ("Sri Venkateshwara Motors", "Shop 7, Main Road, Beside HP Petrol Pump, Hyderabad 500034", "India"),
    ]

    test_templates = [
        # US test entities
        ("Summit Financial Group Corp", "700 Wall St, Suite 500, New York, NY 10005", "US"),
        ("Blue Horizon Seafoods LLC", "250 Ocean Ave, Boston, MA 02110", "US"),
        ("Pinnacle Sports Equipment", "1800 Olympic Blvd, Los Angeles, CA 90064", "US"),
        ("Redwood Dental Clinic", "45 California St, San Francisco, CA 94111", "US"),
        ("Green Leaf Organics Co", "1200 4th Ave, Seattle, WA 98101", "US"),
        # India test entities
        ("Gupta Brothers Sweets Pvt Ltd", "Shop 14, Main Market, Opp Metro Pillar 120, Delhi 110092", "India"),
        ("Anand Auto Spares", "Near City Bus Stand, Station Road, Jaipur 302006", "India"),
        ("Choudhary Marble and Granites", "Plot 55, RIICO Industrial Area, Mansarovar, Jaipur 302020", "India"),
        ("Shree Ram Jewellers", "Shop 8, Zaveri Bazaar, Kalbadevi, Mumbai 400002", "India"),
        ("Mehta Diagnostic Laboratory", "Ground Floor, Behind Apollo Pharmacy, Bannerghatta Rd, Bengaluru 560076", "India"),
        # France test entities (Open-set evaluation!)
        ("Boulangerie Dupont et Fils SARL", "15 Rue de la Republique, 75011 Paris", "France"),
        ("Atelier Mecanique Martin SA", "42 Boulevard Victor Hugo, 69002 Lyon", "France"),
        ("Pharmacie Centrale de Provence", "8 Place de l Hotel de Ville, 13100 Aix-en-Provence", "France"),
        ("Librairie Saint Michel SAS", "27 Rue Saint-Andre des Arts, 75006 Paris", "France"),
        ("Vignobles et Terroirs de Bordeaux", "120 Route des Chateaux, 33250 Pauillac", "France"),
    ]

    def noise_name(name: str) -> str:
        variations = [
            name.replace("Pvt Ltd", "Private Limited"),
            name.replace("Inc", "Incorporated"),
            name.replace("LLC", "L.L.C."),
            name.replace("Corp", "Corporation"),
            name.replace(" and ", " & "),
            name.replace("SARL", "S.A.R.L."),
            name.replace("SA", "S.A."),
            name.replace("SAS", "S.A.S."),
            name.lower(),
            name.upper(),
        ]
        return random.choice(variations)

    def noise_addr(addr: str) -> str:
        variations = [
            addr.replace("Pkwy", "Parkway"),
            addr.replace("Ave", "Avenue"),
            addr.replace("St", "Street"),
            addr.replace("Dr", "Drive"),
            addr.replace("Rd", "Road"),
            addr.replace("Opp", "Opposite"),
            addr.replace("Near", "Nr"),
            addr.replace("Rue", "R."),
            addr.replace("Boulevard", "Bd"),
            addr.replace("Route", "Rte"),
            addr.lower(),
        ]
        return random.choice(variations)

    # 1. Generate Training Data
    s1_rows = []
    s2_rows = []
    s3_rows = []
    gt_rows = []

    s2_id_counter = 1
    s3_id_counter = 1

    for i, (name, addr, country) in enumerate(train_templates, start=1):
        s1_id = f"S1-{i:05d}"
        s1_rows.append([s1_id, name, addr, country])

        # Decide matching structure:
        # 30% singletons, 40% 1 match (S2 or S3), 30% 2 matches (both S2 and S3)
        match_type = random.choice(["single", "s2_only", "s3_only", "both"])
        matched_ids = []

        if match_type in ["s2_only", "both"]:
            s2_id = f"S2-{s2_id_counter:05d}"
            s2_id_counter += 1
            s2_rows.append([s2_id, noise_name(name), noise_addr(addr), country])
            matched_ids.append(s2_id)

        if match_type in ["s3_only", "both"]:
            s3_id = f"S3-{s3_id_counter:05d}"
            s3_id_counter += 1
            s3_rows.append([s3_id, noise_name(name), noise_addr(addr), country])
            matched_ids.append(s3_id)

        gt_rows.append([s1_id, ",".join(matched_ids)])

    # Add some distractor / non-matching entities in S2 and S3
    for k in range(5):
        s2_distractor = f"S2-{s2_id_counter:05d}"
        s2_id_counter += 1
        s2_rows.append([s2_distractor, f"Distractor US Business {k}", f"999 Random Blvd, Chicago, IL 60601", "US"])

        s3_distractor = f"S3-{s3_id_counter:05d}"
        s3_id_counter += 1
        s3_rows.append([s3_distractor, f"Distractor Indian Shop {k}", f"Shop 99, Random Galli, Delhi 110001", "India"])

    # Write train TSVs
    def write_tsv(path, header, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)

    write_tsv(os.path.join(train_dir, "train_source1.tsv"), ["entity_id", "business_name", "business_address", "country"], s1_rows)
    write_tsv(os.path.join(train_dir, "train_source2.tsv"), ["entity_id", "business_name", "business_address", "country"], s2_rows)
    write_tsv(os.path.join(train_dir, "train_source3.tsv"), ["entity_id", "business_name", "business_address", "country"], s3_rows)
    write_tsv(os.path.join(train_dir, "train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"], gt_rows)

    # 2. Generate Test Data (US, India, and France)
    t_s1_rows = []
    t_s2_rows = []
    t_s3_rows = []
    t_s2_cnt = 1
    t_s3_cnt = 1

    for i, (name, addr, country) in enumerate(test_templates, start=101):
        s1_id = f"S1-{i:05d}"
        t_s1_rows.append([s1_id, name, addr, country])

        match_choice = random.choice(["single", "s2", "s3", "both"])
        if match_choice in ["s2", "both"]:
            s2_id = f"S2-{t_s2_cnt:05d}"
            t_s2_cnt += 1
            t_s2_rows.append([s2_id, noise_name(name), noise_addr(addr), country])
        if match_choice in ["s3", "both"]:
            s3_id = f"S3-{t_s3_cnt:05d}"
            t_s3_cnt += 1
            t_s3_rows.append([s3_id, noise_name(name), noise_addr(addr), country])

    # Add test distractors
    t_s2_rows.append([f"S2-{t_s2_cnt:05d}", "Bistro Parisien Non-matching", "12 Rue de Rivoli, 75001 Paris", "France"])
    t_s3_rows.append([f"S3-{t_s3_cnt:05d}", "California Surf Supplies", "100 Coastal Hwy, Malibu, CA 90265", "US"])

    write_tsv(os.path.join(test_dir, "test_source1.tsv"), ["entity_id", "business_name", "business_address", "country"], t_s1_rows)
    write_tsv(os.path.join(test_dir, "test_source2.tsv"), ["entity_id", "business_name", "business_address", "country"], t_s2_rows)
    write_tsv(os.path.join(test_dir, "test_source3.tsv"), ["entity_id", "business_name", "business_address", "country"], t_s3_rows)

    print(f"Generated realistic benchmark datasets in:\n  - {train_dir}\n  - {test_dir}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    generate_sample_datasets(base)
