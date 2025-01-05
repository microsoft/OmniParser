"""
Define some colorful stuffs for better visualization in the chat.
"""

# Define the RGB colors for each letter
colors = {
    'S': 'rgb(106, 158, 210)',
    'h': 'rgb(111, 163, 82)',
    'o': 'rgb(209, 100, 94)',
    'w': 'rgb(238, 171, 106)',
    'U': 'rgb(0, 0, 0)',  
    'I': 'rgb(0, 0, 0)',  
}

# Construct the colorful "ShowUI" word
colorful_text_showui = "**"+''.join(
    f'<span style="color:{colors.get(letter, "black")}">{letter}</span>'
    for letter in "ShowUI"
)+"**"


colorful_text_vlm = "**OmniParser Agent**"

colorful_text_user = "**User**"

# print(f"colorful_text_showui: {colorful_text_showui}")
# **<span style="color:rgb(106, 158, 210)">S</span><span style="color:rgb(111, 163, 82)">h</span><span style="color:rgb(209, 100, 94)">o</span><span style="color:rgb(238, 171, 106)">w</span><span style="color:rgb(0, 0, 0)">U</span><span style="color:rgb(0, 0, 0)">I</span>**