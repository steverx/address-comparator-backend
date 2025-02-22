import { Request, Response } from 'express';
import { findAddressMatches } from '../services/database.service';

export const compareAddresses = async (req: Request, res: Response) => {
    try {
        const { sourceFile, columns, threshold } = req.body;
        const results = [];

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
        res.status(500).json({
            status: 'error',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
};

function formatAddress(row: any, columns: string[]): string {
    return columns.map(col => row[col] || '').filter(Boolean).join(' ');
}