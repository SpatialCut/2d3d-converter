from process_video import processvideo

def test_processvideo():
    # Replace with your local file paths
    local_filename = "ForBiggerMeltdowns.mp4"
    output_filename = "output.mp4"

    try:
        processvideo(local_filename, output_filename)
        print(f"Video processing completed for {local_filename}. Output saved as {output_filename}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    test_processvideo()