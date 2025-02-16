# show number of rules created with no filter: 8 products, 14 baskets, 1,477 rules 
# set max-len=2, and sort by lift to talk about complemantary, substitude, and independent product
# lift higher than 1: complementary: milk and coffee have more than 50% support,
# sort by lift: minimum lift: coffe and tea + appple and pear but which of them we are more confident? high support coffee and tea
# lift at 1: independent product: coffee and meat with 50% support, pear and coffee with 30% support
# by setting min_support = .5, the number of rules reduces to 18
# minimum support = products set bought together is taking care of the most criteria because of closure property of sets
# minimum confidence 
# high lift due to low support (fake), high support-> will not create high lift and not recommendation .
# low number of baskets: minimum support is high , maximum support is low
# add new product sugar?no rule change why!
# add sugar and coffee  
# remove one basket and check the number of rules 
# remove baket 1, remove 2 and remove 1, 2 ?? it does not change the initial number of rules generated but it does change probabilities 
# remove 1,2,3,4,5,7,10,12,14 basket, 6 products and 56 rules with 8 products 1478 rules 
# remove 1,2,3,4,5,7,10,12,14 basket, add sugar and no #rule changes, add sugar and coffee and see for lift 1.5? 
# min support at .01, .07<confidence<1     but min support = .5, .53<confidence< 1  
# min support at .01, .28<lift< 7          but min support = .5, .95<lift< 1.55  
# min support at .6, lift=.969 
# min support at .6 and lift =1 will not give recommendation 

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import matplotlib.pyplot as plt
import qrcode
from io import BytesIO
import sys
st.write(sys.executable) 

#!pip install seaborn
import seaborn as sns

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
    "coffee": "☕",
    "banana": "🍌", 
    "egg": "🥚",
    "strawberry": "🍓",
    "grapes": "🍇",
    "watermelon": "🍉",
    "ice cream": "🍨",
     "fish": "🐟",
    "carrot": "🥕",
    "broccoli": "🥦"
}

# Initial simulated transactional data
data = {
    'Basket': list(range(1, 6)),  # Adjusted for new baskets
    'Items': [
        ["apple", "milk", "fish"],  # Complementary: coffee and milk
        ["beer", "meat", "fish"],  # Independent: meat and pear
        ["tea", "beer", "pear"],  # Invalid: coffee and tea (substitute), will be removed
        ["apple", "fish"],  # No milk or coffee together (substitute condition)
        ["pear", "beer"]
    ]
}
df = pd.DataFrame(data)

# Create a display DataFrame with icons
df_display = df.copy()
df_display['Items'] = df_display['Items'].apply(
    lambda items: ", ".join([f"{product_icons.get(item, '❓')} {item}" for item in items])
)


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
# Modify remove_basket function to accept a list of basket IDs
def remove_basket(basket_ids=None):
    """
    Remove multiple baskets from the transactional data.
    """
    global df, df_display

    if basket_ids is not None:
        # Convert the input string to a list of integers (basket IDs)
        basket_ids_to_remove = [int(basket_id.strip()) for basket_id in basket_ids.split(",")]

        # Remove the baskets specified by the user
        df = df[~df['Basket'].isin(basket_ids_to_remove)].reset_index(drop=True)
        df_display = df_display[~df_display['Basket'].isin(basket_ids_to_remove)].reset_index(drop=True)
    
    return df_display

# Sidebar for removing multiple baskets
st.sidebar.title("Remove Baskets")
# Text input in the sidebar
basket_ids_to_remove = st.sidebar.text_input( "Enter the basket IDs to remove (comma-separated, e.g., '1, 3, 5'):")

if basket_ids_to_remove:
    df_display = remove_basket(basket_ids_to_remove.strip())

#if basket_to_remove != None:
#    df_display = remove_basket(basket_id=basket_to_remove)

# Sidebar for adding new baskets
st.sidebar.title("Create New Basket")
new_basket_input = st.sidebar.text_area(
    "Enter items in your basket (comma-separated, e.g., 'apple, milk, fish, meat, beer, tea, pear, banana, coffee, egg,strawberry, grapes, watermelon, ice cream, fish, carrot, broccoli'). Add one basket at a time:"
)


if new_basket_input:
    df_display = add_new_basket(new_basket_input.strip())
############################## Make table larger
st.title("Transactional Data Table")
#st.dataframe(df_display)
df_display = df_display.rename(columns= {"Items": "Items in the Basket"})
# Convert the dataframe to an HTML table
html_table = df_display.to_html(index=False)

############################
# Add custom CSS for font size and styling
# Define custom CSS to style and set the width of the table and its columns
custom_css = """
    <style>
    .dataframe {
        font-size: 20px !important;
        border-collapse: collapse;
        margin-left: auto;
        margin-right: auto;
    }
    .dataframe th {
        padding: 8px;
        border: 1px solid black;
        text-align: left !important; /* Force left alignment for headers */
    }
    .dataframe td {
        padding: 8px;
        border: 1px solid black;
        text-align: left; /* Left-align data as well */
    }
    </style>
    """

# Combine CSS and HTML
html_code = custom_css + html_table

# Display the HTML table with Streamlit
st.markdown(html_table, unsafe_allow_html=True)
######################################
# Preprocess data into transaction format
def preprocess_transactions(transactions):
    all_items = sorted(set(item.lower() for sublist in transactions for item in sublist))
    return pd.DataFrame([
        {item: (item in map(str.lower, transaction)) for item in all_items}
        for transaction in transactions
    ])

transaction_data = preprocess_transactions(df['Items'])
transaction_data.index = [f"Basket {i+1}" for i in range(len(transaction_data))]

############################ Display frequency of table 
# Frequency table of products (sum of 1s for each product)
product_frequencies = transaction_data.sum().sort_values(ascending=False)

# Plotting the frequency table
st.write("Product Frequency Chart:")
fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size
sns.barplot(x=product_frequencies.index, y=product_frequencies.values, palette="Blues_d", ax=ax)  # Custom color palette
ax.set_title("Product Frequency", fontsize=16, weight='bold')  # Title with custom size and weight
ax.set_xlabel("Products", fontsize=12, weight='bold')  # X-axis label with custom size and weight
ax.set_ylabel("Frequency", fontsize=12, weight='bold')  # Y-axis label with custom size and weight
ax.tick_params(axis='both', labelsize=10)  # Adjust tick label size
ax.grid(True, linestyle='--', alpha=0.7)  # Add gridlines with dashed lines and slight transparency
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
############################
# Sidebar sliders for thresholds
st.sidebar.title("Adjust Thresholds")
#min_antecedent_support = st.sidebar.slider("Min Antecedents Support", 0.0001, 1.0, 0.0001, 0.01)
#min_consequent_support = st.sidebar.slider("Min Consequents Support", 0.0001, 1.0, 0.0001, 0.01)
min_support = st.sidebar.slider("Min Support", 0.0001, 1.0, 0.0001, 0.01)
min_confidence = st.sidebar.slider("Min Confidence", 0.0001, 1.0, 0.0001, 0.01)
#min_lift = st.sidebar.slider("Min Lift", 0.01, 5.0, 0.05, 0.01)
#max_len = st.sidebar.slider("Max Length of Product Combinations", 1, len(transaction_data.columns), len(transaction_data.columns))

frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)

# Generate association rules (only if frequent_itemsets is not empty)
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    # Apply additional filtering
    filtered_rules = rules[
        (rules['support'] >= min_support) &
        (rules['confidence'] >= min_confidence)]
    filtered_rules.sort_values(by=['confidence','support'],inplace=True,ascending=False)

else:
    filtered_rules = pd.DataFrame(columns=['antecedents','consequents', 'support', 'confidence'])

filtered_rules['confidence'] = filtered_rules['confidence'].apply(lambda x: round(x, 3))

# Number of unique products (items) in the transactional data
num_products = len(df['Items'].explode().unique())

# Number of baskets (transactions) in the data
num_baskets = len(df)

# Number of rules after filtering
num_filtered_rules = len(filtered_rules)
###################################
# Display information in Streamlit
st.markdown(f"#### Number of products: {num_products}")
st.markdown(f"#### Number of baskets: {num_baskets}")
st.markdown(f"#### Number of association rules: {num_filtered_rules}")
######################################
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

# Convert the dataframe to an HTML table
html_table = filtered_rules_display[['antecedents', 'consequents',  'support', 'confidence']].to_html(index=False)

# Format rules for display
if not filtered_rules.empty:
    filtered_rules_display = format_rules(filtered_rules.copy())
    st.title("Association Rules")
    #st.dataframe(filtered_rules_display[['antecedents','antecedent support' ,'consequents', 'consequent support', 'support', 'confidence']])
    st.markdown(html_table, unsafe_allow_html=True)
else:
    st.markdown("No association rules found for the given thresholds.")
    #st.markdown(f"#### Number of products: {num_products}")
######################################################## Smart basket suggestions
st.title("Smart Basket Suggestions")
all_antecedents = set(item for rule in filtered_rules['antecedents'] for item in rule)
smart_basket = {f"{product_icons.get(item, '❓')} {item}" for item in all_antecedents}
html_string = f"""
<p style='font-size:30px; color:red;'>
    <strong>Suggested Smart Basket: {', '.join(smart_basket)}</strong>
</p>
"""
# Display the styled HTML
st.markdown(html_string, unsafe_allow_html=True)
