from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator
import instructor
import openai
from pydantic import Field
from typing import List
import os

# API Key setup
API_KEY = ""
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key is not set. Please set the API key as a static variable or in the environment variable OPENAI_API_KEY."
    )


class NutritionLabel(BaseIOSchema):
    """Represents the complete nutritional information from a food label"""

    calories: int = Field(..., description="Calories per serving")
    total_fat: float = Field(..., description="Total fat in grams")
    saturated_fat: float = Field(..., description="Saturated fat in grams")
    trans_fat: float = Field(..., description="Trans fat in grams")
    cholesterol: int = Field(..., description="Cholesterol in milligrams")
    sodium: int = Field(..., description="Sodium in milligrams")
    total_carbohydrates: float = Field(..., description="Total carbohydrates in grams")
    dietary_fiber: float = Field(..., description="Dietary fiber in grams")
    total_sugars: float = Field(..., description="Total sugars in grams")
    added_sugars: float = Field(..., description="Added sugars in grams")
    protein: float = Field(..., description="Protein in grams")
    vitamin_d: float = Field(..., description="Vitamin D in micrograms")
    calcium: int = Field(..., description="Calcium in milligrams")
    iron: float = Field(..., description="Iron in milligrams")
    potassium: int = Field(..., description="Potassium in milligrams")
    serving_size: str = Field(..., description="The size of a single serving of this product")
    servings_per_container: float = Field(..., description="Number of servings contained in the package")
    product_name: str = Field(
        ...,
        description="The full name or description of the type of the food/drink. e.g: 'Coca Cola Light', 'Pepsi Max', 'Smoked Bacon', 'Chianti Wine'",
    )


class NutritionAnalysisInput(BaseIOSchema):
    """Input schema for nutrition label analysis"""

    instruction_text: str = Field(..., description="The instruction for analyzing the nutrition label")
    images: List[instructor.Image] = Field(..., description="The nutrition label images to analyze")


class NutritionAnalysisOutput(BaseIOSchema):
    """Output schema containing extracted nutrition information"""

    analyzed_labels: List[NutritionLabel] = Field(
        ..., description="List of nutrition labels extracted from the provided images"
    )


# Configure the nutrition analysis system
nutrition_analyzer = AtomicAgent[NutritionAnalysisInput, NutritionAnalysisOutput](
    config=AgentConfig(
        client=instructor.from_openai(openai.OpenAI(api_key=API_KEY)),
        model="gpt-5.6-terra",
        model_api_parameters={"reasoning_effort": "low"},
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a specialized nutrition label analyzer.",
                "You excel at extracting precise nutritional information from food label images.",
                "You understand various serving size formats and measurement units.",
                "You can process multiple nutrition labels simultaneously.",
            ],
            steps=[
                "For each nutrition label image:",
                "1. Locate and identify the nutrition facts panel",
                "2. Extract all serving information and nutritional values",
                "3. Validate measurements and units for accuracy",
                "4. Compile the nutrition facts into structured data",
            ],
            output_instructions=[
                "For each analyzed nutrition label:",
                "1. Record complete serving size information",
                "2. Extract all nutrient values with correct units",
                "3. Ensure all measurements are properly converted",
                "4. Include all extracted labels in the final result",
            ],
        ),
    )
)


def main():
    print("Starting nutrition label analysis...")

    # Construct the path to the test images
    script_directory = os.path.dirname(os.path.abspath(__file__))
    test_images_directory = os.path.join(os.path.dirname(script_directory), "test_images")
    image_path_1 = os.path.join(test_images_directory, "nutrition_label_1.png")
    image_path_2 = os.path.join(test_images_directory, "nutrition_label_2.jpg")
    # Create and submit the analysis request
    analysis_request = NutritionAnalysisInput(
        instruction_text="Please analyze these nutrition labels and extract all nutritional information.",
        images=[instructor.Image.from_path(image_path_1), instructor.Image.from_path(image_path_2)],
    )

    try:
        # Process the nutrition labels
        print("Analyzing nutrition labels...")
        analysis_result = nutrition_analyzer.run(analysis_request)
        print("Analysis completed successfully")

        # Display the results
        for i, label in enumerate(analysis_result.analyzed_labels, 1):
            print(f"\nNutrition Label {i}:")
            print(f"Product Name: {label.product_name}")
            print(f"Serving Size: {label.serving_size}")
            print(f"Servings Per Container: {label.servings_per_container}")
            print(f"Calories: {label.calories}")
            print(f"Total Fat: {label.total_fat}g")
            print(f"Saturated Fat: {label.saturated_fat}g")
            print(f"Trans Fat: {label.trans_fat}g")
            print(f"Cholesterol: {label.cholesterol}mg")
            print(f"Sodium: {label.sodium}mg")
            print(f"Total Carbohydrates: {label.total_carbohydrates}g")
            print(f"Dietary Fiber: {label.dietary_fiber}g")
            print(f"Total Sugars: {label.total_sugars}g")
            print(f"Added Sugars: {label.added_sugars}g")
            print(f"Protein: {label.protein}g")
            print(f"Vitamin D: {label.vitamin_d}mcg")
            print(f"Calcium: {label.calcium}mg")
            print(f"Iron: {label.iron}mg")
            print(f"Potassium: {label.potassium}mg")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
