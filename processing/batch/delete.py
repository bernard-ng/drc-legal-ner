from openai import OpenAI

client = OpenAI()


def delete_all_files():
    try:
        limit, after, deleted_count = 100, None, 0
        while True:
            files = client.files.list(limit=limit, after=after)
            if not files.data:
                print("No files found to delete.")
                break

            for file in files.data:
                try:
                    print(f"Deleting file: {file.id}")
                    client.files.delete(file.id)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting file {file.id}: {e}")
                    continue

            # Handle pagination
            if files.has_next_page():
                next_page_info = files.next_page_info()
                if next_page_info:
                    after = next_page_info.params.get("after")
                else:
                    break
            else:
                break

        print(f"Total {deleted_count} files have been deleted.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    confirmation = input("Are you sure you want to delete all files? (yes/no): ")
    if confirmation.lower() == "yes":
        delete_all_files()
    else:
        print("Operation cancelled.")
