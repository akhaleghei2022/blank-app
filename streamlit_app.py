import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import qrcode
from io import BytesIO

# Input for QR code URL
url = st.text_input("Enter the text or URL for which you want to generate a QR code:")

if url:
    # Generate the QR code
    qr = qrcode.make(url)
    
    # Save the QR code image to a BytesIO object
    img_buffer = BytesIO()
    qr.save(img_buffer)
    img_buffer.seek(0)  # Go back to the beginning of the BytesIO buffer
    
    # Display the QR code image in Streamlit
    st.image(img_buffer, use_column_width=True)
    
# Product icon mapping
product_icons = {
    "apple": "🍎",
    "beer": "🍺",
    "rice": "🍚",
    "meat": "🥩",
    "pear": "🍐",
    "milk": "🥛",
    "tea": "🍵",
    "coffee": "☕"
}

# Initial simulated transactional data
data = {
    'Basket': list(range(1, 9)),
    'Items': [
        ["apple", "beer", "rice", "meat"],
        ["apple", "beer", "rice"],
        ["apple", "beer"],
        ["apple", "pear"],
        ["milk", "beer", "rice", "meat"],
        ["milk", "beer", "rice"],
        ["milk", "beer"],
        ["milk", "pear"]
    ]
}
df = pd.DataFrame(data)

# Create a display DataFrame with icons
df_display = df.copy()
df_display['Items'] = df_display['Items'].apply(
    lambda items: ", ".join([f"{product_icons.get(item, '❓')} {item}" for item in items])
)

#st.dataframe(df_display)

# Add new basket
def add_new_basket(new_baskets_input):
    """
    Add multiple new baskets to the transactional data.
    """
    global df, df_display
    
    # Split the new basket input into individual baskets
    baskets = new_baskets_input.strip().split("\n")
    for basket_input in baskets:
        new_items = {item.strip().lower() for item in basket_input.split(',')}
        df = pd.concat([df, pd.DataFrame({'Basket': [len(df) + 1], 'Items': [list(new_items)]})], ignore_index=True)
        
        # Update the display DataFrame
        new_items_with_icons = ", ".join([f"{product_icons.get(item, '❓')} {item}" for item in new_items])
        df_display.loc[len(df_display)] = [len(df_display) + 1, new_items_with_icons]

    return df_display

# Remove basket
# Remove basket with a default option if invalid input
def remove_basket(basket_id=None):
    """
    Remove a basket from the transactional data.
    If basket_id is None or invalid, remove the first basket with valid data.
    """
    global df, df_display

    # If the user didn't provide a valid basket_id, set the default basket to remove
    if basket_id is not None and basket_id > len(df):
        # Find the first basket that doesn't have null values
        basket_to_remove = df['Basket'].iloc[0]
    else:
        basket_to_remove = basket_id

    # Remove the basket
    df = df[df['Basket'] != basket_to_remove].reset_index(drop=True)
    df_display = df_display[df_display['Basket'] != basket_to_remove].reset_index(drop=True)
    
    return df_display


# Sidebar for adding new baskets
st.sidebar.title("Create New Basket")
new_basket_input = st.sidebar.text_area(
    "Enter items in your basket (comma-separated, e.g., 'Milk, Tea, Coffee'). Add one basket at a time:"
)

if new_basket_input:
    df_display = add_new_basket(new_basket_input.strip())

# Sidebar for removing a basket
st.sidebar.title("Remove Basket")
basket_to_remove = st.sidebar.number_input(
    "Enter the basket ID to remove (leave empty for the first basket):", 
    min_value=-1, max_value=len(df), step=1, value=None
)

if basket_to_remove != None:
    df_display = remove_basket(basket_id=basket_to_remove)


st.title("Transactional Data Table")
st.dataframe(df_display)

# Preprocess data into transaction format
def preprocess_transactions(transactions):
    all_items = sorted(set(item.lower() for sublist in transactions for item in sublist))
    return pd.DataFrame([
        {item: (item in map(str.lower, transaction)) for item in all_items}
        for transaction in transactions
    ])

transaction_data = preprocess_transactions(df['Items'])
transaction_data.index = [f"Basket {i+1}" for i in range(len(transaction_data))]

# Sidebar sliders for thresholds
st.sidebar.title("Adjust Thresholds")
min_support = st.sidebar.slider("Min Support", 0.0, 1.0, 0.5, 0.05)
min_confidence = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
min_lift = st.sidebar.slider("Min Lift", 0.0, 5.0, 0.05, 0.05)

# Generate frequent itemsets and association rules
frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
filtered_rules = rules[
    (rules['support'] >= min_support) &
    (rules['confidence'] >= min_confidence) &
    (rules['lift'] >= min_lift)
]

# Format rules for display
def format_rules(rules_df):
    rules_df['antecedents'] = rules_df['antecedents'].apply(
        lambda x: ", ".join([f"{product_icons.get(item, '❓')} {item}" for item in x])
    )
    rules_df['consequents'] = rules_df['consequents'].apply(
        lambda x: ", ".join([f"{product_icons.get(item, '❓')} {item}" for item in x])
    )
    return rules_df

filtered_rules_display = format_rules(filtered_rules.copy())

st.title("Filtered Association Rules")
st.dataframe(filtered_rules_display[['antecedents', 'consequents','consequent support', 'support', 'confidence', 'lift']])

# Smart basket suggestions
st.title("Smart Basket Suggestions")
all_antecedents = set(item for rule in filtered_rules['antecedents'] for item in rule)
smart_basket = {f"{product_icons.get(item, '❓')} {item}" for item in all_antecedents}
st.markdown(f"**<span style='color:red'>Suggested Smart Basket: {', '.join(smart_basket)}</span>**", unsafe_allow_html=True)

# Forgotten item suggestions
st.title("Check for Forgotten Items in Your Basket")
forgotten_basket_input = st.text_input("Enter the items already in your basket (comma-separated):")

if forgotten_basket_input:
    user_items = {item.strip().lower() for item in forgotten_basket_input.split(',')}
    forgotten_items = set()
    for antecedent in filtered_rules['antecedents']:
        if not antecedent.issubset(user_items):
            forgotten_items.update(antecedent - user_items)

    forgotten_items_with_icons = {f"{product_icons.get(item, '❓')} {item}" for item in forgotten_items}
    if forgotten_items:
        st.markdown(f"**<span style='color:red'>Forgotten items: {', '.join(forgotten_items_with_icons)}</span>**", unsafe_allow_html=True)
    else:
        st.write("No forgotten items found.")
