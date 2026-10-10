"""
Generate a small, self-contained sample dataset of phishing and legitimate
emails so the project trains and runs fully offline, with no downloads.

The corpus is *composed from fragments*, so every email is unique rather than
a repeated template, and the two classes deliberately share vocabulary
("verify", "account", "confirm", "click here", "limited time offer"). A small
amount of label noise is injected. Together this keeps the reported metrics in
a realistic range instead of a trivial 100%.

This is a *demo* dataset. For a real deployment, replace data/emails.csv with
a public corpus (Enron-Spam, SpamAssassin, Nazario phishing corpus, or the
Kaggle "Phishing Email Detection" dataset) and keep the same two columns:
    label, text
where label = 1 for phishing and 0 for legitimate.

Run:
    python scripts/generate_sample_data.py
"""

import os
import csv
import random

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "data", "emails.csv")

LABEL_NOISE_RATE = 0.03

# ------------------------------------------------------------------ phishing ---
PHISH_GREETINGS = [
    "Dear Customer", "Dear valued member", "Dear user", "Dear account holder",
    "Hello", "Dear Sir/Madam", "Attention",
]
PHISH_OPENERS = [
    "we detected unusual activity on your account",
    "your account has been temporarily locked",
    "our security team flagged a suspicious sign-in",
    "your recent payment could not be processed",
    "we were unable to verify your billing information",
    "your subscription is about to expire",
    "you are eligible for a refund",
    "your mailbox storage is almost full",
    "a sign-in from an unrecognized device was detected",
    "your account will be suspended within 24 hours",
    "your parcel is being held at customs",
    "your invoice is overdue and requires immediate attention",
]
PHISH_ASKS = [
    "please confirm your username and password using the link below",
    "verify your identity by entering your card number and PIN",
    "click here to update your billing details and reactivate your account",
    "kindly provide your login credentials so we can unlock your account",
    "submit your bank account and social security number to receive the funds",
    "confirm your details now to avoid permanent closure",
    "re-enter your login details to view your statement",
    "pay the release fee within 48 hours or the item will be returned",
]
PHISH_URGENCY = [
    "This is your final notice.",
    "Act immediately to avoid suspension.",
    "Offer expires in 12 hours.",
    "Failure to respond will result in account closure.",
    "You must act within 24 hours.",
    "",
]
PHISH_LINKS = [
    "http://secure-verify-login.xyz/confirm",
    "http://{brand}-account-security.top/verify",
    "http://billing-update-center.info/pay",
    "http://claim-your-prize.click/winner",
    "http://account-alerts.ru/secure",
    "http://192.168.44.7/login.php",
    "http://bit.ly/3xR9q2k",
]

# ---------------------------------------------------------------- legitimate ---
# Business, personal, HR, transactional and automated notifications. Several use
# the same "verify / confirm / click" vocabulary as phishing on purpose.
LEGIT_GREETINGS = [
    "Hi team", "Hello", "Dear customer", "Hi Alex", "Hello all",
    "Dear subscriber", "Hi",
]
LEGIT_OPENERS = [
    "here are the notes from this morning's standup",
    "your order has shipped and will arrive in 3-5 business days",
    "attached is your monthly account statement for the period ending the 30th",
    "following up on our conversation about the API refactor",
    "your interview is confirmed for Thursday at 3:00 PM",
    "your pull request has been reviewed and approved",
    "thank you for subscribing to our weekly newsletter",
    "timesheets are due by end of day Friday",
    "just confirming our lunch meeting tomorrow at 1pm",
    "thanks for the update, I'll take a look this afternoon",
    "the build passed on CI and the release is now live",
    "here is the agenda for tomorrow's design review",
    "a new device signed in to your account",
    "your payment was received and your subscription is active",
    "welcome aboard - here is everything you need for your first week",
]
LEGIT_ASKS = [
    "please confirm your email address to activate your subscription",
    "click here to view the full issue on our blog",
    "you can update your account preferences from the dashboard",
    "verify your delivery address so we can ship your order",
    "click here to read the release notes",
    "log in to your account to download the invoice",
    "review the doc before Thursday's planning session",
    "let me know if a quick call works",
    "if this was you, no action is needed",
]
LEGIT_NOTES = [
    "No action is required.",
    "Thanks, Priya.",
    "The whole team did great work.",
    "Reach out to HR if you need access.",
    "Nice work on the test coverage.",
    "",
]
LEGIT_LINKS = [
    "https://www.example-shop.com/track/58231",
    "https://newsletter.example.com/issues/142",
    "https://app.example.com/dashboard",
    "https://example.com/orders/58231",
]

BRANDS = ["paypal", "netflix", "amazon", "microsoft", "apple", "bank", "dhl", "fedex"]

# Short emails of both classes, so the model learns that a short message is not
# automatically suspicious (and that short messages can also be phishing).
SHORT_LEGIT = [
    "Thanks for the update, I'll take a look this afternoon.",
    "Can you send over the latest version of the deck?",
    "Just confirming our lunch tomorrow at 1pm.",
    "Congratulations on the launch!",
    "Reminder: dentist appointment on Tuesday at 10am.",
    "Great work on the presentation yesterday.",
    "See you at the offsite next week.",
    "Sounds good, let's sync up on Friday.",
    "Here's the agenda for tomorrow's review.",
    "Happy birthday! Hope you have a great day.",
    "The report is ready, I've shared it in the drive folder.",
    "No problem, I'll handle it.",
]
SHORT_PHISH = [
    "Your account will be closed. Confirm your password now.",
    "Urgent: verify your account at http://secure-login.xyz",
    "You have won a prize! Claim it at http://claim-prize.click",
    "Payment failed. Update your card at http://billing.top/pay",
    "Final notice: click http://account-alerts.ru to avoid suspension.",
    "Reset your password immediately at http://reset-verify.info",
    "Your parcel is held. Pay at http://dhl-fee.click",
]


def compose(parts):
    return " ".join(p for p in parts if p).strip()


def make_phishing():
    if random.random() < 0.25:
        return random.choice(SHORT_PHISH)
    brand = random.choice(BRANDS)
    text = compose([
        random.choice(PHISH_GREETINGS) + ",",
        random.choice(PHISH_OPENERS).capitalize() + ".",
        random.choice(PHISH_ASKS).capitalize() + ".",
        random.choice(PHISH_URGENCY),
        "Link: " + random.choice(PHISH_LINKS).format(brand=brand),
    ])
    return text


def make_legit():
    if random.random() < 0.25:
        return random.choice(SHORT_LEGIT)
    text = compose([
        random.choice(LEGIT_GREETINGS) + ",",
        random.choice(LEGIT_OPENERS).capitalize() + ".",
        random.choice(LEGIT_ASKS).capitalize() + ".",
        random.choice(LEGIT_NOTES),
        "Link: " + random.choice(LEGIT_LINKS),
    ])
    return text


def main(n_per_class=900):
    rows = []
    for _ in range(n_per_class):
        label = 0 if random.random() < LABEL_NOISE_RATE else 1
        rows.append((label, make_phishing()))
    for _ in range(n_per_class):
        label = 1 if random.random() < LABEL_NOISE_RATE else 0
        rows.append((label, make_legit()))

    random.shuffle(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {os.path.normpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
