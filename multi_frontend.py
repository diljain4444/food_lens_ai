# import streamlit as st
# from PIL import Image
# import io
# from multi_back import extract_information

# st.title("🍲 Ingredient Detector")

# # ── File uploader ──────────────────────────────────────────────────────────────
# uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

# if uploaded_file is None:
#     st.info("Please upload an image to get started.")
#     st.stop()

# image = Image.open(uploaded_file)
# st.image(image, caption="Uploaded Image")

# # ── Call API only once per unique image ───────────────────────────────────────
# image_key = f"{uploaded_file.name}_{uploaded_file.size}"

# if st.session_state.get("image_key") != image_key:
#     with st.spinner("Detecting ingredients..."):
#         buf = io.BytesIO()
#         image.save(buf, format="JPEG")
#         image_bytes = buf.getvalue()
#         try:
#             st.session_state.ingredients = extract_information(image_bytes)
#             st.session_state.image_key = image_key
#             # ✅ reset checkbox states when new image is uploaded
#             st.session_state.checked = {
#                 ing: True for ing in st.session_state.ingredients
#             }
#         except Exception as e:
#             st.error(f"API error: {e}")
#             st.stop()

# # ✅ Initialize checked dict if not present
# if "checked" not in st.session_state:
#     st.session_state.checked = {
#         ing: True for ing in st.session_state.get("ingredients", [])
#     }

# # ── Ingredient checkboxes ──────────────────────────────────────────────────────
# st.subheader("Detected Ingredients")
# st.caption("Uncheck any ingredient you want to remove.")

# for ingredient in st.session_state.ingredients:
#     # ✅ read/write checkbox state directly in session_state
#     current = st.session_state.checked.get(ingredient, True)
#     result = st.checkbox(ingredient, value=current, key=f"cb_{ingredient}")
#     st.session_state.checked[ingredient] = result  # ✅ persist the state

# # ── Add custom ingredient ──────────────────────────────────────────────────────
# st.divider()
# col1, col2 = st.columns([3, 1])
# with col1:
#     new_item = st.text_input("Add a missing ingredient",
#                               label_visibility="collapsed",
#                               placeholder="Type ingredient name...")
# with col2:
#     if st.button("➕ Add", use_container_width=True):
#         name = new_item.strip()
#         if name and name not in st.session_state.ingredients:
#             st.session_state.ingredients.append(name)
#             st.session_state.checked[name] = True  # ✅ new item checked by default
#             st.rerun()
#         elif not name:
#             st.warning("Enter a name first.")
#         else:
#             st.warning("Already in the list.")

# # ── Confirm final list ─────────────────────────────────────────────────────────
# st.divider()
# if st.button("✅ Confirm Ingredients", type="primary"):
#     # ✅ build final list directly from checked dict — no rerun dependency
#     st.session_state.final_ingredients = [
#         ing for ing in st.session_state.ingredients
#         if st.session_state.checked.get(ing, True)
#     ]

# if "final_ingredients" in st.session_state:
#     st.success("Final ingredient list confirmed!")
#     st.write(st.session_state.final_ingredients)
#     final_list = st.session_state.final_ingredients


from PIL import Image
image = Image.open(
    r"C:\Users\DIL JAIN\Desktop\agentic\multimodal_rag\Top_10_Protein_Rich_Indian_Foods_Feature_img.jpg"
)

from multi_back import extract_information,workflow

# text=extract_information(image)
text=["tomato","onion","paneer","apple","banana","ginger","milk","cashew","almonds","garam-masala","wheat-flour","rice"]
result=workflow.invoke({"initial_info":text})
print(result)
