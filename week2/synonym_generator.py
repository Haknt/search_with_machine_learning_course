import argparse
import fasttext

def get_synonyms(model, word, threshold):
    neighbors = model.get_nearest_neighbors(word)
    synonyms = []
    for similarity, neighbor in neighbors:
        if similarity >= threshold:
            synonyms.append(neighbor)
    return synonyms

def main(model_path, top_words_path, output_path, threshold):
    model = fasttext.load_model(model_path)

    with open(top_words_path, 'r') as file, open(output_path, 'w') as outfile:
        for line in file:
            word = line.strip()
            synonyms = get_synonyms(model, word, threshold)
            outfile.write(word + ',' + ','.join(synonyms) + '\n')
    
    print(f"Synonyms generated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/workspace/datasets/fasttext/title_model.bin")
    parser.add_argument("--top_words_path", type=str, default="/workspace/datasets/fasttext/top_words.txt")
    parser.add_argument("--output_path", type=str, default="/workspace/datasets/fasttext/synonyms.csv")
    parser.add_argument("--threshold", type=float, default=0.75)

    args = parser.parse_args()
    main(args.model_path, args.top_words_path, args.output_path, args.threshold)