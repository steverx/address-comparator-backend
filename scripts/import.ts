import { importCsvToDatabase } from '../utils/csvImport';
import path from 'path';

const csvFilePath = process.argv[2];

if (!csvFilePath) {
    console.error('Please provide a CSV file path');
    process.exit(1);
}

const absolutePath = path.resolve(csvFilePath);

importCsvToDatabase(absolutePath)
    .then(() => {
        console.log('Import completed successfully');
        process.exit(0);
    })
    .catch((error) => {
        console.error('Import failed:', error);
        process.exit(1);
    });