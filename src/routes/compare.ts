import { Router } from 'express';
import { z } from 'zod';
import { findAddressMatches } from '../services/database.service';

const router = Router();

const CompareRequestSchema = z.object({
    sourceFile: z.array(z.record(z.string())),
    columns: z.array(z.string()),
    threshold: z.number().min(0).max(100)
});

router.post('/compare', async (req, res) => {
    try {
        const validated = CompareRequestSchema.parse(req.body);
        const results = await Promise.all(
            validated.sourceFile.map(async (row) => {
                const address = validated.columns
                    .map(col => row[col])
                    .filter(Boolean)
                    .join(' ');
                const matches = await findAddressMatches(address, validated.threshold);
                return { original_row: row, matches };
            })
        );
        res.json({ status: 'success', data: results });
    } catch (error) {
        res.status(400).json({ 
            status: 'error', 
            message: error instanceof Error ? error.message : 'Invalid request' 
        });
    }
});

export default router;