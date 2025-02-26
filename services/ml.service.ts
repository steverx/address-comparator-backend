import { TfidfVectorizer } from '../utils/vectorizer';
import { cosineSimilarity } from '../utils/similarity';

export class AddressMatcher {
    private static instance: AddressMatcher;
    private vectorizer: TfidfVectorizer;
    private addressVectors: number[][];

    static async getInstance(): Promise<AddressMatcher> {
        if (!AddressMatcher.instance) {
            AddressMatcher.instance = new AddressMatcher();
            await AddressMatcher.instance.initialize();
        }
        return AddressMatcher.instance;
    }

    async findMatches(queryAddress: string, options: {
        addresses: string[],
        threshold: number,
        returnTop: number
    }): Promise<Array<{ index: number, score: number }>> {
        const queryVector = this.vectorizer.transform([queryAddress])[0];
        const similarities = options.addresses.map((_, index) => ({
            index,
            score: cosineSimilarity(queryVector, this.addressVectors[index])
        }));

        return similarities
            .filter(result => result.score >= options.threshold)
            .sort((a, b) => b.score - a.score)
            .slice(0, options.returnTop);
    }

    private async initialize() {
        this.vectorizer = new TfidfVectorizer({
            ngramRange: [2, 3],
            analyzer: 'char'
        });
    }
}