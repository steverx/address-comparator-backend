import { findAddressMatches } from '../services/database.service';
import { Router, Request, Response } from 'express';

const router = Router();

function formatAddress(row: any, columns: any): string {
    // Implement the logic to format the address based on the row and columns
    return `${row[columns.street]} ${row[columns.city]} ${row[columns.state]} ${row[columns.zip]}`;
}

router.post('/compare', async (req: Request, res: Response) => {
    try {
        const { sourceFile, columns, threshold } = req.body;
        const results: { original_row: any; matches: any[] }[] = [];

        for (const row of sourceFile) {
            const address = formatAddress(row, columns);
            const matches = await findAddressMatches(address, threshold / 100);

            if (matches.length > 0) {
                results.push({
                    original_row: row,
                    matches: matches
                });
            }
        }

        res.json({
            status: 'success',
            data: results
        });
    } catch (error) {
        console.error('Comparison error:', error);
        res.status(500).json({
            status: 'error',
            error: error.message
        });
    }
});