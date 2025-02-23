import natural from 'natural';

export class TfidfVectorizer {
    private tfidf: natural.TfIdf;
    private documents: string[];

    constructor(options?: { ngramRange?: [number, number], analyzer?: string }) {
        this.tfidf = new natural.TfIdf();
        this.documents = [];
    }

    fit(documents: string[]) {
        this.documents = documents;
        documents.forEach(document => {
            this.tfidf.addDocument(document);
        });
    }

    transform(documents: string[]): number[][] {
        return documents.map(document => {
            const vector: number[] = [];
            this.tfidf.listTerms(document, 100).forEach(item => {
                vector.push(item.tfidf);
            });
            return vector;
        });
    }

    fitTransform(documents: string[]): number[][] {
        this.fit(documents);
        return this.transform(documents);
    }
}