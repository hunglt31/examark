import os
import concurrent.futures
from multiple_choice_detection import write_result_to_csv, process_image
from support_functions import straighten_image


if __name__ == "__main__":
    cut = int(input("""Straighten images? 
                Yes - 1 
                No - 2
                Your choice (integer): """))
    if cut == 1:
        exam_folder_path = '../exams/LTHDT_all'
        straightened_image_folder_path = '../straightened_exams/LTHDT_all'
        os.makedirs(straightened_image_folder_path, exist_ok=True)
        for image in os.listdir(exam_folder_path):
            image_path = os.path.join(exam_folder_path, image)
            image_name = image.split('.')[0]
            straighten_image(image_path, image_name, straightened_image_folder_path)

    check = int(input("""Run app? 
                    Yes - 1
                    No - 2
                    Your choice (integer): """))
    
    if check == 1:                
        straightened_image_folder_path = '../straightened_exams/LTHDT_all'
        path_result = '../exam_results/LTHDT_all.csv'

        results = []
        time_limit = 300
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            
            for image in os.listdir(straightened_image_folder_path):
                image_path = os.path.join(straightened_image_folder_path, image)
                future = executor.submit(process_image, image_path, path_result)
                futures[future] = image_path
            
            for future in concurrent.futures.as_completed(futures):
                image_path = futures[future]
                try:
                    exam_result = future.result(timeout=time_limit)
                    if exam_result:  
                        results.append(exam_result)
                except concurrent.futures.TimeoutError:
                    print(f"Processing of {image_path} timed out after {time_limit} seconds.")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        write_result_to_csv(results, path_result)
    